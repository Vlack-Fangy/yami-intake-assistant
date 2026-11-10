"""
On-demand lifecycle for the vLLM model server.

WHY THIS EXISTS
---------------
vLLM pins a whole A100 for as long as it runs, which blocked other people on the
box even when nobody was talking to Yami. So the GPU now follows the user:
started on /start, released after IDLE_TIMEOUT_S of silence.

The *bot* process (bot.py) still runs 24x7 — it has to, because Telegram only
delivers /start to a process that is polling. With LLM_BACKEND=vllm the bot
loads no model at all (see llm.warmup_model), so an idle bot costs ~100MB of RAM
and zero GPU. The thing that gets torn down is the vLLM server, which is the
part that was actually in anyone's way.

THE CARDINAL RULE
-----------------
Never disturb anyone else's work. Every ambiguity here resolves to "don't use
the GPU":

  * A candidate GPU must have ZERO compute processes AND near-zero used memory
    AND near-zero utilization — sampled TWICE, seconds apart, both clean.
  * If nvidia-smi is missing, errors, or returns something we can't parse, we
    refuse to start. No guessing.
  * If a foreign process appears on our GPU while we are still booting, we abort
    and get off the card — they were there first as far as we're concerned.
  * We only ever kill processes that are ours (same unix user, vllm cmdline, on
    the GPU we claimed).

Priority order is 3, 2, 1, 0 — highest index first, matching the convention on
this box that the low GPUs are the busy ones.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Awaitable, Callable, Optional

from utils import get_logger

logger = get_logger("gpu_manager")


# ── Configuration ───────────────────────────────────────────────────────────
# Master switch. Set LLM_LAZY_GPU=0 to go back to the old "vLLM is always up,
# started by watchdog.sh" behaviour.
ENABLED = os.getenv("LLM_LAZY_GPU", "0") not in ("0", "false", "False", "")

# Same default as llm.VLLM_BASE_URL (llm.py:47) — read from env rather than
# importing llm, to keep this module importable on its own.
VLLM_BASE_URL = os.getenv("LLM_VLLM_BASE_URL", "http://127.0.0.1:8100/v1").rstrip("/")

TMUX_SESSION = os.getenv("VLLM_TMUX_SESSION", "yami-vllm")
SERVE_SCRIPT = os.getenv("VLLM_SERVE_SCRIPT", "scripts/serve_vllm.sh")

GPU_PRIORITY = [
    int(x) for x in os.getenv("GPU_PRIORITY", "3,2,1,0").split(",") if x.strip() != ""
]

# ── Dedicated-card mode ─────────────────────────────────────────────────────
# Set GPU_DEDICATED_INDEX=<n> when a specific card is ours by policy rather than
# by us proving it is idle. That inverts the question the scan below asks:
#
#   shared box   → "is this card pristine?"   (nobody else is on it)
#   dedicated    → "is there ROOM on it?"     (co-tenants are expected)
#
# The pristine test is wrong on a dedicated card. Ours has other jobs of our own
# on it — and, in practice, other people's too. Requiring zero processes there
# means _find_free_gpu never returns anything and every patient is told "every
# GPU is busy" forever, which is a bricked bot, not a safety guarantee.
#
# What does NOT change: we still never touch a process we cannot prove is ours.
# On a co-tenanted card that rule matters MORE, not less — it is the only thing
# standing between us and someone else's three-day training run.
def _optional_int(name: str) -> Optional[int]:
    raw = os.getenv(name, "").strip()
    if raw == "":
        return None
    try:
        return int(raw)
    except ValueError:
        logger.error(f"{name}={raw!r} is not an integer — ignoring")
        return None


GPU_DEDICATED_INDEX = _optional_int("GPU_DEDICATED_INDEX")
DEDICATED = GPU_DEDICATED_INDEX is not None

# Free VRAM the card must still have before we launch. Must exceed what vLLM
# will actually reserve (VLLM_GPU_MEM_UTIL x total in scripts/serve_vllm.sh),
# or we start a server that OOMs on load. Default 55000 MiB gives real margin
# over the 0.35 x 143771 MiB = ~50320 MiB reservation on an H200 NVL.
VLLM_REQUIRED_MIB = int(os.getenv("VLLM_REQUIRED_MIB", "55000"))

# A truly idle A100 with no display attached reports single-digit MiB. 200 is
# generous headroom for driver/ECC overhead without masking a real process.
GPU_FREE_MEM_MIB = int(os.getenv("GPU_FREE_MEM_MIB", "200"))
GPU_FREE_UTIL_PCT = int(os.getenv("GPU_FREE_UTIL_PCT", "5"))
# Two samples this far apart must BOTH be clean. Catches a card between kernels
# of someone's training loop.
GPU_RECHECK_DELAY_S = float(os.getenv("GPU_RECHECK_DELAY_S", "3"))

STARTUP_TIMEOUT_S = float(os.getenv("VLLM_STARTUP_TIMEOUT_S", "720"))   # 12 min
HEALTH_POLL_S = float(os.getenv("VLLM_HEALTH_POLL_S", "10"))
SHUTDOWN_GRACE_S = float(os.getenv("VLLM_SHUTDOWN_GRACE_S", "45"))

# LLM_IDLE_TIMEOUT_S=0 means "never release" — see idle_check. The 15-minute
# release exists to hand a shared card back to other people; on a card that is
# ours by policy it buys nothing and costs the next patient a multi-minute cold
# start, so warm is the right default there.
IDLE_TIMEOUT_S = float(os.getenv("LLM_IDLE_TIMEOUT_S", "900"))    # 15 min
IDLE_WARN_S = float(os.getenv("LLM_IDLE_WARN_S", "780"))          # warn at 13 min
WARM = IDLE_TIMEOUT_S <= 0


class Status(Enum):
    READY = "ready"
    NO_FREE_GPU = "no_free_gpu"      # transient — someone else is using them all
    ERROR = "error"                  # broken — needs a human


@dataclass
class _State:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    gpu_index: Optional[int] = None
    started_at: Optional[float] = None
    # Boot window, used to spot messages typed while we were still coming up.
    booting: bool = False
    boot_started_at: Optional[float] = None
    ready_at: Optional[float] = None
    last_activity: float = field(default_factory=time.time)
    turns_in_flight: int = 0
    warned_idle: bool = False
    # chat_ids that have touched the model since the last boot, so we can tell
    # them when the GPU goes away.
    chats: set = field(default_factory=set)


_state = _State()


# ── nvidia-smi plumbing ─────────────────────────────────────────────────────
def _smi(query: str, kind: str = "gpu") -> Optional[list[list[str]]]:
    """
    Run one nvidia-smi query. Returns parsed rows, or None on ANY problem —
    callers must treat None as "we don't know, so don't touch the GPU".
    """
    if not shutil.which("nvidia-smi"):
        logger.error("nvidia-smi not found on PATH")
        return None
    flag = "--query-gpu" if kind == "gpu" else "--query-compute-apps"
    try:
        out = subprocess.run(
            ["nvidia-smi", f"{flag}={query}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
    except Exception as error:            # timeout, OSError, ...
        logger.error(f"nvidia-smi {flag}={query} failed: {error}")
        return None
    if out.returncode != 0:
        logger.error(f"nvidia-smi {flag}={query} rc={out.returncode}: {out.stderr.strip()[:300]}")
        return None
    rows = []
    for line in out.stdout.splitlines():
        line = line.strip()
        if line:
            rows.append([cell.strip() for cell in line.split(",")])
    return rows


def _compute_apps() -> Optional[dict[str, list[str]]]:
    """
    {gpu key -> [pid, ...]} for every compute process on the box.

    Driver versions differ on which identifier they expose, so we try gpu_uuid
    first and fall back to gpu_bus_id. If neither works we return None and the
    caller refuses to start — an unreadable process list is exactly the case
    where we must not assume a card is free.
    """
    for field_name in ("gpu_uuid", "gpu_bus_id"):
        rows = _smi(f"{field_name},pid", kind="apps")
        if rows is None:
            continue
        apps: dict[str, list[str]] = {}
        for row in rows:
            if len(row) >= 2:
                apps.setdefault(row[0], []).append(row[1])
        return apps
    logger.error("could not read compute-app list from nvidia-smi (tried gpu_uuid and gpu_bus_id)")
    return None


def _snapshot() -> Optional[dict[int, dict]]:
    """One reading of every GPU: {index: {...}}. None if anything is unreadable."""
    rows = _smi("index,uuid,pci.bus_id,memory.used,utilization.gpu,memory.total")
    if rows is None:
        return None
    apps = _compute_apps()
    if apps is None:
        return None

    gpus: dict[int, dict] = {}
    for row in rows:
        if len(row) < 6:
            logger.error(f"unparseable nvidia-smi row: {row!r}")
            return None
        try:
            index = int(row[0])
            mem_used = int(float(row[3]))
            util = int(float(row[4]))
            mem_total = int(float(row[5]))
        except ValueError:
            logger.error(f"unparseable nvidia-smi row: {row!r}")
            return None
        uuid, bus_id = row[1], row[2]
        gpus[index] = {
            "uuid": uuid,
            "bus_id": bus_id,
            "mem_used": mem_used,
            "mem_total": mem_total,
            "mem_free": max(0, mem_total - mem_used),
            "util": util,
            "pids": apps.get(uuid, []) or apps.get(bus_id, []),
        }
    return gpus


def _detect_our_gpu() -> Optional[int]:
    """
    Which GPU is our vLLM session sitting on? None if we can't tell.

    Needed because the bot restarts more often than the model server does. The
    watchdog respawns `python bot.py` 15s after any crash, and that fresh process
    has no memory of the GPU it claimed last time — but the tmux session and the
    engine are still very much alive on a card. Without re-adopting it, shutdown
    would kill the session and then skip the "is the card actually clean?"
    verification, which is the check that protects everyone else on the box.
    """
    tree = _our_process_tree()
    if tree is None:
        return None
    snapshot = _snapshot()
    if snapshot is None:
        return None
    for index, gpu in snapshot.items():
        if any(pid in tree for pid in gpu["pids"]):
            return index
    return None


def _is_free(gpu: dict) -> bool:
    return (
        not gpu["pids"]
        and gpu["mem_used"] <= GPU_FREE_MEM_MIB
        and gpu["util"] <= GPU_FREE_UTIL_PCT
    )


def _has_headroom(gpu: dict) -> bool:
    """Enough free VRAM for vLLM, regardless of who else is on the card.

    The dedicated-mode gate. Deliberately ignores process count and utilisation:
    on a card that is ours by policy, a busy neighbour is not a reason to refuse
    service — only a card with no room left is.
    """
    return gpu["mem_free"] >= VLLM_REQUIRED_MIB


def _find_dedicated_gpu() -> tuple[Optional[int], Status, str]:
    """The pinned card, if it currently has room. No scan, no priority list.

    One sample, not two: the two-sample recheck exists to catch a card that is
    between kernels of somebody's training loop, which only matters when we are
    trying to prove a card is UNUSED. We are not — we are checking for free
    memory on a card we already own, and vLLM's own allocation is what settles
    the race anyway.
    """
    snapshot = _snapshot()
    if snapshot is None:
        return None, Status.ERROR, "cannot read GPU state from nvidia-smi"

    index = GPU_DEDICATED_INDEX
    gpu = snapshot.get(index)
    if gpu is None:
        # Misconfiguration, not congestion — retrying will never fix it.
        return None, Status.ERROR, (
            f"GPU_DEDICATED_INDEX={index} does not exist "
            f"(visible: {sorted(snapshot)})"
        )

    if not _has_headroom(gpu):
        detail = (
            f"GPU {index} has {gpu['mem_free']} MiB free, need {VLLM_REQUIRED_MIB} "
            f"({gpu['mem_used']}/{gpu['mem_total']} MiB used by "
            f"{len(gpu['pids'])} process(es))"
        )
        logger.info(f"dedicated GPU {index} has no room — {detail}")
        return None, Status.NO_FREE_GPU, detail

    logger.info(
        f"claiming dedicated GPU {index} "
        f"({gpu['mem_free']} MiB free of {gpu['mem_total']}, "
        f"{len(gpu['pids'])} co-tenant process(es))"
    )
    return index, Status.READY, f"GPU {index} has {gpu['mem_free']} MiB free"


def _find_free_gpu() -> tuple[Optional[int], Status, str]:
    """
    First fully-idle GPU in priority order, confirmed by two samples taken
    GPU_RECHECK_DELAY_S apart.

    Returns (index, status, detail). A None index means don't start anything —
    the status separates "everything is busy, come back later" (the user should
    retry) from "we cannot see the GPUs at all" (a human needs to look).
    """
    if DEDICATED:
        return _find_dedicated_gpu()

    first = _snapshot()
    if first is None:
        return None, Status.ERROR, "cannot read GPU state from nvidia-smi"
    candidates = [i for i in GPU_PRIORITY if i in first and _is_free(first[i])]
    if not candidates:
        busy = ", ".join(
            f"GPU{i}: {first[i]['mem_used']}MiB/{first[i]['util']}%/{len(first[i]['pids'])}proc"
            for i in sorted(first)
        )
        logger.info(f"no fully-free GPU — {busy}")
        return None, Status.NO_FREE_GPU, "every GPU is in use"

    time.sleep(GPU_RECHECK_DELAY_S)
    second = _snapshot()
    if second is None:
        return None, Status.ERROR, "cannot read GPU state from nvidia-smi"
    for index in candidates:
        if index in second and _is_free(second[index]):
            logger.info(f"claiming GPU {index} (idle in both samples)")
            return index, Status.READY, f"GPU {index} idle in both samples"
        logger.info(f"GPU {index} looked free then became busy — skipping")
    return None, Status.NO_FREE_GPU, "candidate GPUs became busy on re-check"


def _our_process_tree() -> Optional[set[str]]:
    """
    Every PID descended from our tmux pane — i.e. every process that is ours.
    None means we could not work it out.

    Ancestry, NOT the command line. The obvious test ("does the cmdline say
    vllm?") is wrong: vLLM V1 spawns its EngineCore worker through
    multiprocessing, so the process actually holding the VRAM has a cmdline of
    `python -c from multiprocessing.spawn import spawn_main ...` with no mention
    of vllm anywhere. That made us mistake our own engine for someone else's job
    and abandon a perfectly good GPU seconds after claiming it.
    """
    try:
        panes = subprocess.run(
            ["tmux", "list-panes", "-t", TMUX_SESSION, "-F", "#{pane_pid}"],
            capture_output=True, text=True, timeout=10,
        )
        if panes.returncode != 0:
            return None
        roots = [line.strip() for line in panes.stdout.splitlines() if line.strip()]
        if not roots:
            return None

        table = subprocess.run(
            ["ps", "-eo", "pid=,ppid="], capture_output=True, text=True, timeout=15,
        )
        if table.returncode != 0:
            return None
    except Exception as error:
        logger.error(f"could not map our process tree: {error}")
        return None

    children: dict[str, list[str]] = {}
    for line in table.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2:
            children.setdefault(parts[1], []).append(parts[0])

    tree, stack = set(roots), list(roots)
    while stack:
        for child in children.get(stack.pop(), []):
            if child not in tree:
                tree.add(child)
                stack.append(child)
    return tree


def _pids_on(gpu_index: int) -> tuple[list[str], list[str], bool]:
    """
    (ours, foreign, certain) for the compute processes on gpu_index.

    `certain` is False when we could not establish ownership. Callers must treat
    that as "don't know" rather than "foreign" — guessing wrong in that direction
    is what made us abandon our own GPUs.
    """
    snapshot = _snapshot()
    if snapshot is None or gpu_index not in snapshot:
        return [], [], False
    pids = snapshot[gpu_index]["pids"]
    tree = _our_process_tree()
    if tree is None:
        return [], [], False
    ours = [pid for pid in pids if pid in tree]
    foreign = [pid for pid in pids if pid not in tree]
    return ours, foreign, True


# ── vLLM process control ────────────────────────────────────────────────────
async def is_healthy(timeout: float = 5.0) -> bool:
    """Does the vLLM server answer /v1/models?"""
    import httpx
    try:
        # trust_env=False for the same reason as llm._call_vllm: this box exports
        # http_proxy, and routing a localhost call through Squid returns an HTML
        # error page that would read as "server down" while it is perfectly fine.
        async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
            resp = await client.get(f"{VLLM_BASE_URL}/models")
            return resp.status_code == 200
    except Exception:
        return False


def _tmux_alive() -> bool:
    try:
        return subprocess.run(
            ["tmux", "has-session", "-t", TMUX_SESSION],
            capture_output=True, timeout=10,
        ).returncode == 0
    except Exception:
        return False


def _launch(gpu_index: int) -> bool:
    """
    Start vLLM in its own tmux session on exactly one GPU.

    Deliberately NOT wrapped in a restart loop (unlike the old watchdog.sh): if
    it dies we want the failure to surface to the user, not a crash loop
    hammering a card we may no longer be entitled to.
    """
    env_prefix = f"CUDA_VISIBLE_DEVICES={gpu_index}"
    command = f"{env_prefix} bash {SERVE_SCRIPT} 2>&1 | tee -a logs/vllm.out"
    try:
        result = subprocess.run(
            ["tmux", "new-session", "-d", "-s", TMUX_SESSION, command],
            capture_output=True, text=True, timeout=30,
        )
    except Exception as error:
        logger.error(f"tmux new-session failed: {error}")
        return False
    if result.returncode != 0:
        logger.error(f"tmux new-session rc={result.returncode}: {result.stderr.strip()[:300]}")
        return False
    logger.info(f"launched vLLM on GPU {gpu_index} in tmux session {TMUX_SESSION}")
    return True


def _kill(gpu_index: Optional[int]) -> None:
    """
    Stop vLLM and make sure the card is actually released.

    Killing the tmux session normally suffices; the PID sweep afterwards is
    insurance against a detached child holding VRAM, because a GPU we *think*
    we released but haven't is precisely the thing that ruins someone's day.

    Success is judged by the CARD being clean on a shared box, because that is
    the property other people actually care about there. On a DEDICATED card the
    test has to be "our processes are gone" instead: co-tenants mean the card is
    never pristine, so the old test would never pass — we would SIGKILL-sweep on
    every single shutdown and then log a "needs manual inspection" alarm about a
    neighbour who is doing nothing wrong.
    """
    # Snapshot our PIDs BEFORE killing the pane: once tmux is gone we can no
    # longer derive ownership from ancestry, and we must never SIGKILL a process
    # we cannot prove is ours.
    ours: list[str] = []
    if gpu_index is not None:
        ours, _foreign, _certain = _pids_on(gpu_index)

    try:
        subprocess.run(["tmux", "kill-session", "-t", TMUX_SESSION],
                       capture_output=True, timeout=15)
    except Exception as error:
        logger.error(f"tmux kill-session failed: {error}")

    if gpu_index is None:
        return

    def _released(snapshot: Optional[dict]) -> bool:
        """Have we let go of the card, by whichever test applies here?"""
        if snapshot is None or gpu_index not in snapshot:
            return False
        if not DEDICATED:
            return _is_free(snapshot[gpu_index])
        # Dedicated: only OUR pids need to be gone. `ours` was captured while
        # tmux still existed, so it is the authoritative list.
        still_there = set(snapshot[gpu_index]["pids"]) & set(ours)
        return not still_there

    deadline = time.time() + SHUTDOWN_GRACE_S
    while time.time() < deadline:
        if _released(_snapshot()):
            logger.info(f"GPU {gpu_index} released")
            return
        time.sleep(2)

    for pid in ours:
        logger.error(f"pid {pid} still on GPU {gpu_index} after "
                     f"{SHUTDOWN_GRACE_S}s — SIGKILL")
        try:
            subprocess.run(["kill", "-9", pid], capture_output=True, timeout=10)
        except Exception as error:
            logger.error(f"kill -9 {pid} failed: {error}")

    time.sleep(3)
    if not _released(_snapshot()):
        # Loud, because someone else is now looking at a card we are squatting on.
        what = "our processes are still on it" if DEDICATED else "it is still not clean"
        logger.error(f"GPU {gpu_index} not released after shutdown — {what}; "
                     f"needs manual inspection (nvidia-smi)")


# ── Public API ──────────────────────────────────────────────────────────────
ProgressCb = Callable[[str], Awaitable[None]]


async def preflight() -> tuple[Status, str]:
    """
    Cheap, non-blocking answer to "could we serve this person right now?".
    Safe to call on /start before committing to anything.
    """
    if not ENABLED:
        return Status.READY, ""
    if await is_healthy():
        return Status.READY, ""
    gpu, status, detail = await asyncio.to_thread(_find_free_gpu)
    if gpu is None:
        return status, detail
    return Status.READY, detail


async def ensure_ready(progress: Optional[ProgressCb] = None) -> tuple[Status, str]:
    """
    Bring the model up if needed and wait for it. Idempotent and safe to call
    concurrently — the lock means one boot, and everyone else waits on it.

    `progress` is called with a human-readable status line roughly every
    HEALTH_POLL_S so the caller can keep a Telegram message updated.
    """
    if not ENABLED:
        return Status.READY, ""

    async with _state.lock:
        if await is_healthy():
            # A server we did not start in THIS process — we were restarted while
            # it kept running. Adopt it so the idle timer and shutdown know which
            # card to verify.
            if _state.gpu_index is None:
                adopted = await asyncio.to_thread(_detect_our_gpu)
                if adopted is not None:
                    logger.info(f"adopting running vLLM on GPU {adopted} "
                                f"(bot restarted while it was up)")
                    _state.gpu_index = adopted
                    _state.ready_at = _state.ready_at or time.time()
            return Status.READY, ""

        # Stale session with a dead server inside it — clear it out first, or
        # tmux new-session below fails with "duplicate session".
        if _tmux_alive():
            logger.info("tmux session exists but vLLM is not answering — recycling")
            await asyncio.to_thread(_kill, _state.gpu_index)

        gpu_index, status, detail = await asyncio.to_thread(_find_free_gpu)
        if gpu_index is None:
            return status, detail

        if not await asyncio.to_thread(_launch, gpu_index):
            return Status.ERROR, "could not launch the vLLM server"

        _state.gpu_index = gpu_index
        _state.started_at = time.time()
        _state.boot_started_at = _state.started_at
        _state.booting = True
        _state.warned_idle = False

        try:
            return await _await_healthy(gpu_index, progress)
        finally:
            _state.booting = False


async def _await_healthy(gpu_index: int, progress: Optional[ProgressCb]) -> tuple[Status, str]:
    """Poll a freshly launched server until it answers, dies, or times out."""
    deadline = time.time() + STARTUP_TIMEOUT_S
    while time.time() < deadline:
        await asyncio.sleep(HEALTH_POLL_S)

        if await is_healthy():
            elapsed = int(time.time() - _state.started_at)
            logger.info(f"vLLM ready on GPU {gpu_index} after {elapsed}s")
            _state.ready_at = time.time()
            _state.last_activity = _state.ready_at
            return Status.READY, f"ready in {elapsed}s"

        if not _tmux_alive():
            logger.error("vLLM session died during startup — see logs/vllm.out")
            await asyncio.to_thread(_kill, gpu_index)
            _state.gpu_index = None
            return Status.ERROR, "the model server exited while starting"

        # Somebody else landed on this card while we were loading. They get
        # it; we back off rather than compete for VRAM. Only acted on when
        # ownership is CERTAIN — "can't tell" must not evict our own engine.
        #
        # Skipped entirely on a dedicated card. There, co-tenants are the normal
        # state rather than an intrusion: backing off would abandon a card we are
        # entitled to every time a neighbour's job happens to be running, which
        # on a busy box is always. Headroom was checked before launch and vLLM
        # has already reserved its share by this point.
        if not DEDICATED:
            _ours, foreign, certain = await asyncio.to_thread(_pids_on, gpu_index)
            if certain and foreign:
                logger.info(f"foreign process {foreign} appeared on GPU {gpu_index} "
                            f"during startup — backing off")
                await asyncio.to_thread(_kill, gpu_index)
                _state.gpu_index = None
                return Status.NO_FREE_GPU, "another job took the GPU while we were starting"

        if progress:
            elapsed = int(time.time() - _state.started_at)
            try:
                await progress(f"{elapsed // 60}m {elapsed % 60}s")
            except Exception:
                pass    # a failed message edit must never abort a boot

    logger.error(f"vLLM did not become healthy within {STARTUP_TIMEOUT_S}s")
    await asyncio.to_thread(_kill, gpu_index)
    _state.gpu_index = None
    return Status.ERROR, "the model server did not come up in time"


def was_sent_during_boot(sent_at: float) -> bool:
    """
    Was a message (epoch seconds, from Telegram) typed while we were booting?

    Such messages are impatience — "is it ready?", "hello?" — not answers, and
    feeding them to the model derails the interview. The message that TRIGGERS a
    boot is deliberately excluded: it predates boot_started_at, and it is a real
    answer that the patient expects us to act on once we wake up.
    """
    start = _state.boot_started_at
    if start is None or sent_at < start:
        return False
    if _state.booting:
        return True
    return _state.ready_at is not None and sent_at < _state.ready_at


async def shutdown(reason: str = "idle") -> None:
    """Release the GPU. Safe to call when nothing is running."""
    async with _state.lock:
        gpu_index = _state.gpu_index
        if gpu_index is None:
            if not _tmux_alive():
                return
            # Orphaned session from a previous bot process — find its card so the
            # post-kill "card is clean" check can actually run.
            gpu_index = await asyncio.to_thread(_detect_our_gpu)
        logger.info(f"shutting down vLLM ({reason})")
        await asyncio.to_thread(_kill, gpu_index)
        _state.gpu_index = None
        _state.started_at = None
        _state.chats.clear()


def note_activity(chat_id: Optional[int] = None) -> None:
    _state.last_activity = time.time()
    _state.warned_idle = False
    if chat_id is not None:
        _state.chats.add(chat_id)


def begin_turn() -> None:
    _state.turns_in_flight += 1
    _state.last_activity = time.time()


def end_turn() -> None:
    _state.turns_in_flight = max(0, _state.turns_in_flight - 1)
    _state.last_activity = time.time()


def is_running() -> bool:
    return _state.gpu_index is not None or _tmux_alive()


def status_line() -> str:
    if not is_running():
        return "model server: off (starts on demand)"
    idle = int(time.time() - _state.last_activity)
    return (f"model server: on GPU {_state.gpu_index}, idle {idle // 60}m {idle % 60}s, "
            f"{_state.turns_in_flight} turn(s) in flight")


async def idle_check(bot) -> None:
    """
    Called on a timer from bot.py. Warns at IDLE_WARN_S, releases the GPU at
    IDLE_TIMEOUT_S. Never fires while a turn is being generated.

    Does nothing in warm mode (LLM_IDLE_TIMEOUT_S=0). Releasing is a courtesy to
    other people waiting for the card; when the card is ours by policy there is
    nobody to be courteous to, and the only effect is that the next patient pays
    a multi-minute cold start and gets told the bot went to sleep.
    """
    if not ENABLED or WARM or not is_running():
        return
    if _state.turns_in_flight > 0:
        return

    idle = time.time() - _state.last_activity

    if idle >= IDLE_TIMEOUT_S:
        chats = list(_state.chats)
        await shutdown("idle timeout")
        for chat_id in chats:
            try:
                await bot.send_message(
                    chat_id,
                    "I've gone to sleep to free up the server GPU. "
                    "Just send me a message whenever you're ready — I'll wake back up "
                    "(that takes a few minutes) and we'll carry on where we left off.",
                    parse_mode=None,
                )
            except Exception as error:
                logger.error(f"idle-shutdown notice to {chat_id} failed: {error}")
        return

    if idle >= IDLE_WARN_S and not _state.warned_idle:
        _state.warned_idle = True
        remaining = int((IDLE_TIMEOUT_S - idle) / 60) or 1
        for chat_id in list(_state.chats):
            try:
                await bot.send_message(
                    chat_id,
                    f"Still there? If I don't hear from you in about {remaining} more minute(s) "
                    "I'll shut down to free the server. Nothing is lost — message me any time "
                    "and we'll pick up where we left off.",
                    parse_mode=None,
                )
            except Exception as error:
                logger.error(f"idle warning to {chat_id} failed: {error}")


# ── Dry run ─────────────────────────────────────────────────────────────────
# `python gpu_manager.py` prints what we can see and which GPU we WOULD take,
# without starting anything. Read-only — safe to run on a busy server at any
# time, and the first thing to try when someone reports the bot won't start.
if __name__ == "__main__":
    print(f"lazy GPU mode : {'on' if ENABLED else 'off (LLM_LAZY_GPU=0)'}")
    if DEDICATED:
        print(f"mode          : DEDICATED — pinned to GPU {GPU_DEDICATED_INDEX}, no scan")
        print(f"usable means  : >={VLLM_REQUIRED_MIB} MiB free (co-tenants allowed)")
    else:
        print(f"mode          : shared box — scan in priority order")
        print(f"priority      : {GPU_PRIORITY}")
        print(f"free means    : 0 processes, <={GPU_FREE_MEM_MIB} MiB, <={GPU_FREE_UTIL_PCT}% "
              f"(twice, {GPU_RECHECK_DELAY_S}s apart)")
    if WARM:
        print("idle shutdown : never (warm mode, LLM_IDLE_TIMEOUT_S=0)")
    else:
        print(f"idle shutdown : {int(IDLE_TIMEOUT_S / 60)} min (warn at {int(IDLE_WARN_S / 60)} min)")
    print()

    snap = _snapshot()
    if snap is None:
        print("CANNOT READ GPU STATE - the bot would refuse to start and report an error.")
        raise SystemExit(1)

    for index in sorted(snap):
        gpu = snap[index]
        if DEDICATED and index == GPU_DEDICATED_INDEX:
            verdict = "USABLE" if _has_headroom(gpu) else "FULL"
        elif DEDICATED:
            verdict = "-"          # not ours to consider
        else:
            verdict = "FREE" if _is_free(gpu) else "busy"
        why = [f"{gpu['mem_free']} MiB free of {gpu['mem_total']}"]
        if gpu["pids"]:
            why.append(f"{len(gpu['pids'])} process(es): {','.join(gpu['pids'])}")
        if gpu["util"] > GPU_FREE_UTIL_PCT:
            why.append(f"{gpu['util']}% util")
        marker = " <- pinned" if DEDICATED and index == GPU_DEDICATED_INDEX else ""
        print(f"  GPU {index}: {verdict:6}  {'; '.join(why)}{marker}")

    print()
    chosen, status, detail = _find_free_gpu()
    if chosen is None:
        print(f"WOULD NOT START — {status.value}: {detail}")
        if status is Status.NO_FREE_GPU:
            print("  Patient sees: 'every GPU is busy, try again in a few minutes'")
        else:
            print("  Patient sees: 'contact the administrator'")
    else:
        print(f"WOULD CLAIM GPU {chosen} ({detail})")

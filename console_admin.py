"""
Yami Admin Console
Terminal dashboard for Ops — browse patient sessions, inspect progress,
extracted fields, and summaries, pull full transcripts or clinical reports.

Read-only against the existing REST API. Zero backend changes — every view
here is built from endpoints that already exist (/admin/sessions, /fields,
/progress, /summaries, /transcript, /report).

Usage:
    python console_admin.py
    python console_admin.py --host http://localhost:5000
"""

import os
os.environ["no_proxy"] = "localhost,127.0.0.1"
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

import hashlib
import requests
import sys
import select
import time
import argparse
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True)

# ── Config ────────────────────────────────────────────────────

BASE = os.getenv("API_BASE", "http://localhost:5000")

# Same bearer-key treatment as console_client — see the note there.
API_KEY = os.getenv("API_KEY", "")
_client = requests.Session()
if API_KEY:
    _client.headers["Authorization"] = f"Bearer {API_KEY}"
requests = _client
WIDTH = 100  # wider than console_client.py's 70 — this is a data browser, not a chat
REFRESH_SECONDS = 2

# ── ANSI colours (same palette as console_client.py) ───────────

R  = "\033[0m"
B  = "\033[1m"
D  = "\033[2m"
GR = "\033[32m"
YL = "\033[33m"
CY = "\033[36m"
RD = "\033[31m"
MG = "\033[35m"
BL = "\033[34m"

STATUS_COLOR = {"active": GR, "paused": YL, "completed": CY, "crisis": RD}
STATUS_ICON  = {"active": "●", "paused": "⏸", "completed": "✓", "crisis": "🆘"}


# ── HTTP helper ──────────────────────────────────────────────

def get(path: str) -> dict:
    r = requests.get(f"{BASE}{path}")
    r.raise_for_status()
    return r.json()


# ── Display helpers ──────────────────────────────────────────

CLEAR_CMD = "cls" if os.name == "nt" else "clear"
BELL = "\a"


def read_line_with_timeout(prompt: str, timeout: float):
    """
    Like input(), but gives up after `timeout` seconds and returns None
    instead of blocking forever — that's what lets the list view redraw on
    a timer while still accepting a typed command at any moment.

    Unix-only (relies on select() on stdin). Not a concern here, but worth
    knowing if this ever needs to run on Windows.

    Rough edge, worth knowing: if you're mid-way through typing a number
    when the timeout fires, the screen clears and your partial input goes
    with it — the terminal hasn't handed it to us yet (it's still sitting
    in line-buffered input), so there's nothing to preserve. Just retype.
    Not solving this properly without dropping line-input entirely for raw
    single-keypress mode, which is a bigger redesign than "add a refresh".
    """
    print(prompt, end="", flush=True)
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    if not ready:
        return None
    line = sys.stdin.readline()
    if line == "":  # stdin closed (EOF)
        raise EOFError
    return line.rstrip("\n")


def hr(char: str = "─", color: str = D, width: int = None, indent: int = 0):
    print(f"{' ' * indent}{color}{char * (width or WIDTH)}{R}")


def header():
    print()
    hr("═", MG)
    title = "Yami Admin  —  Session Browser"
    pad = (WIDTH - len(title)) // 2
    print(f"{MG}{' ' * pad}{B}{title}{R}")
    hr("═", MG)
    print()


def section_title(text: str):
    print(f"{B}  {text}{R}")


# ── Session list ─────────────────────────────────────────────

# Single source of truth for the session-list table — header and rows are
# both built from this, so they can't drift apart, and columns are always
# joined with an explicit space (content that exactly fills a column still
# gets a guaranteed gap, instead of running into the next column).
LIST_COLUMNS = [("#", 4), ("session", 14), ("patient", 16), ("lang", 6),
                ("section", 20), ("status", 13), ("last active", 19)]
TABLE_WIDTH = sum(w for _, w in LIST_COLUMNS) + (len(LIST_COLUMNS) - 1)


def fetch_sessions(status_filter=None) -> list:
    qs = f"?status={status_filter}&limit=50" if status_filter else "?limit=50"
    return get(f"/admin/sessions{qs}")["sessions"]


def _pad(value: str, width: int) -> str:
    return f"{value:<{width}}"


def print_session_list(sessions: list):
    print()
    hr("─", D, width=TABLE_WIDTH, indent=2)
    header_cells = [_pad(label, w) for label, w in LIST_COLUMNS]
    print(f"  {D}{' '.join(header_cells)}{R}")
    hr("─", D, width=TABLE_WIDTH, indent=2)
    if not sessions:
        print(f"  {D}(no sessions match this filter){R}")
    w_num, w_sid, w_name, w_lang, w_sect, w_status, w_last = [w for _, w in LIST_COLUMNS]
    for i, s in enumerate(sessions, 1):
        status = s.get("status", "—")
        color  = STATUS_COLOR.get(status, D)
        icon   = STATUS_ICON.get(status, "·")
        cells = [
            f"{YL}{_pad(f'[{i}]', w_num)}{R}",
            _pad(s["session_id"][:12], w_sid),
            _pad((s.get("patient_name") or "—")[:14], w_name),
            _pad(s.get("preferred_language") or "—", w_lang),
            _pad((s.get("current_section") or "—").replace("_", " ")[:18], w_sect),
            f"{color}{_pad(f'{icon} {status}', w_status)}{R}",
            f"{D}{_pad((s.get('last_active') or '—')[:19], w_last)}{R}",
        ]
        print(f"  {' '.join(cells)}")
    hr("─", D, width=TABLE_WIDTH, indent=2)
    crisis_count = sum(1 for s in sessions if s.get("status") == "crisis")
    if crisis_count:
        print(f"  {RD}{B}⚠ {crisis_count} session(s) in crisis status{R}")
    print()


# ── Session detail sub-views ─────────────────────────────────

def show_progress(session_id: str):
    data = get(f"/sessions/{session_id}/progress")
    section_title("Progress")
    for s in data["sections"]:
        icon = {"in_progress": "⏳", "complete": "✅", "pending": "🔜"}.get(s["status"], "·")
        pct  = s["completion_pct"]
        bar  = "█" * (pct // 10) + "░" * (10 - pct // 10)
        print(f"  {icon} {s['display_name']:<25} {bar} {pct:>3}%")
    print(f"  {D}Est. remaining: ~{data['total_estimated_mins_remaining']} min{R}")
    print()


def show_fields(session_id: str):
    data = get(f"/sessions/{session_id}/fields")
    section_title("Collected fields")
    if not data:
        print(f"  {D}(none yet){R}")
    for section, fields in data.items():
        filled = {k: v for k, v in fields.items() if v}
        if filled:
            print(f"  {MG}{section}{R}")
            for k, v in filled.items():
                print(f"    {D}{k.replace('_', ' ').title():<25}{R} {v}")
    print()


def show_summaries(session_id: str):
    data = get(f"/sessions/{session_id}/summaries")["summaries"]
    section_title("Section summaries")
    if not data:
        print(f"  {D}(none generated yet){R}")
    for section, text in data.items():
        print(f"  {MG}{section}{R}")
        print(f"    {text}")
        print()


def print_session_header(session: dict):
    print()
    hr("─", CY)
    print(f"  {B}{session['session_id']}{R}")
    name = (session.get("patient_name") or "—")[:20]
    section_name = (session["current_section"] or "—").replace("_", " ")[:20]
    started = (session.get("started_at") or "—")[:20]
    print(f"  {D}patient:{R}  {name:<20}  {D}language:{R} {session['preferred_language']}")
    print(f"  {D}section:{R}  {section_name:<20}  "
          f"{D}status:{R}   {STATUS_COLOR.get(session['status'], D)}{session['status']}{R}")
    print(f"  {D}started:{R}  {started:<20}  {D}active:{R}   {session.get('last_active') or '—'}")
    hr("─", CY)
    print()

    if session["status"] == "crisis":
        print(f"  {RD}{B}🆘 This session is in CRISIS status.{R}")
        print()


def show_detail(session_id: str):
    session = get(f"/sessions/{session_id}")
    print_session_header(session)
    show_progress(session_id)
    show_fields(session_id)
    show_summaries(session_id)

    while True:
        try:
            cmd = input(f"  {D}[t]ranscript  [r]eport  [b]ack  [q]uit  {R}> ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print()
            return
        if cmd == "b":
            return
        elif cmd == "q":
            sys.exit(0)
        elif cmd == "t":
            data = get(f"/sessions/{session_id}/transcript")
            print()
            print(data["transcript"])
            print()
        elif cmd == "r":
            data = get(f"/sessions/{session_id}/report")
            print()
            print(data["report"])
            print()
        else:
            print(f"  {D}unknown command{R}")


# ── Main loop ────────────────────────────────────────────────

FILTER_MAP = {
    "a": None, "all": None,
    "act": "active", "active": "active",
    "pau": "paused", "paused": "paused",
    "comp": "completed", "completed": "completed",
    "cri": "crisis", "crisis": "crisis",
}


def run():
    header()
    try:
        health = get("/health")
        model_ok = "ready" if health.get("model_loaded") else f"{RD}NOT loaded{R}"
        db_ok = "ok" if health.get("db_reachable") else f"{RD}DOWN{R}"
        print(f"  {GR}{D}── ✓ Connected  |  model {model_ok}{D}  |  db {db_ok}{R}")
        time.sleep(0.6)  # let the connection line actually be readable before the first clear
    except Exception as e:
        print(f"  {RD}✗ Cannot connect to API at {BASE}: {e}{R}")
        print(f"  {D}Start server with: uvicorn api:app --port 5000{R}")
        sys.exit(1)

    status_filter = None
    known_crisis_ids = set()

    while True:
        try:
            sessions = fetch_sessions(status_filter)
        except Exception as e:
            sessions = []
            fetch_error = str(e)
        else:
            fetch_error = None

        current_crisis_ids = {s["session_id"] for s in sessions if s.get("status") == "crisis"}
        new_crisis_ids = current_crisis_ids - known_crisis_ids
        known_crisis_ids = current_crisis_ids

        os.system(CLEAR_CMD)
        header()
        if fetch_error:
            print(f"  {RD}Could not fetch sessions: {fetch_error}{R}")
        print_session_list(sessions)

        if new_crisis_ids:
            print(f"{BELL}  {RD}{B}🆘 NEW crisis session detected just now{R}")
            print()

        filt_label = status_filter or "all"
        now = datetime.now().strftime("%H:%M:%S")
        print(f"  {D}last refreshed {now}  ·  auto-refresh every {REFRESH_SECONDS}s{R}")
        sys.stdout.flush()
        prompt = (f"  {D}view: {filt_label}  |  [#] open  "
                  f"[a]ll [act]ive [pau]sed [comp]leted [cri]sis  [q]uit  {R}> ")

        try:
            choice = read_line_with_timeout(prompt, REFRESH_SECONDS)
        except (KeyboardInterrupt, EOFError):
            print()
            break

        if choice is None:
            continue  # timed out — loop back around and refresh

        choice = choice.strip().lower()

        if choice == "q":
            break
        elif choice in FILTER_MAP:
            status_filter = FILTER_MAP[choice]
        elif choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(sessions):
                try:
                    show_detail(sessions[idx]["session_id"])
                except SystemExit:
                    raise
                except Exception as e:
                    print(f"  {RD}Error loading session: {e}{R}")
            else:
                print(f"  {D}no session #{choice}{R}")
        elif choice != "":
            print(f"  {D}unknown command{R}")

    print(f"  {D}Goodbye{R}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=BASE, help="API base URL")
    parser.add_argument("--refresh", type=float, default=REFRESH_SECONDS, help="Auto-refresh interval in seconds")
    args = parser.parse_args()
    BASE = args.host
    REFRESH_SECONDS = args.refresh
    run()

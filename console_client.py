"""
Yami Console Client
Interactive chat interface using the Yami REST API.
Simulates end-game chatbot experience in terminal.

Usage:
    python console_client.py
    python console_client.py --host http://localhost:5000
"""

import os
os.environ["no_proxy"] = "localhost,127.0.0.1"
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

import requests
import uuid
import sys
import time
import textwrap
import argparse
import hashlib

# ── Config ────────────────────────────────────────────────────

BASE = os.getenv("API_BASE", "http://localhost:5000")

# The API requires `Authorization: Bearer <key>` on every request. Rebinding the
# module name to a Session carrying the header means the helpers below need no
# server ignores is harmless, so this always sends one.
API_KEY = os.getenv("API_KEY", "")
_client = requests.Session()
if API_KEY:
    _client.headers["Authorization"] = f"Bearer {API_KEY}"
requests = _client
WIDTH = 70

# ── ANSI colours ──────────────────────────────────────────────

R  = "\033[0m"       # reset
B  = "\033[1m"       # bold
D  = "\033[2m"       # dim
GR = "\033[32m"      # green
YL = "\033[33m"      # yellow
CY = "\033[36m"      # cyan
RD = "\033[31m"      # red
MG = "\033[35m"      # magenta
BL = "\033[34m"      # blue


# ── HTTP helpers ──────────────────────────────────────────────

def get(path: str) -> dict:
    r = requests.get(f"{BASE}{path}")
    r.raise_for_status()
    return r.json()

def post(path: str, body: dict = None) -> dict:
    r = requests.post(f"{BASE}{path}", json=body or {})
    r.raise_for_status()
    return r.json()

def delete(path: str) -> dict:
    r = requests.delete(f"{BASE}{path}")
    r.raise_for_status()
    return r.json()

def patch(path: str, body: dict) -> dict:
    r = requests.patch(f"{BASE}{path}", json=body)
    r.raise_for_status()
    return r.json()


# ── Display helpers ───────────────────────────────────────────

def clear_line():
    print("\r" + " " * WIDTH, end="\r")

def hr(char="─", color=D):
    print(f"{color}{char * WIDTH}{R}")

def header():
    print()
    hr("═", CY)
    title = "Yami  —  Pre-consultation Assistant"
    pad = (WIDTH - len(title)) // 2
    print(f"{CY}{' ' * pad}{B}{title}{R}")
    hr("═", CY)
    print()

def print_bot(message: str):
    """Print bot message with left-aligned label."""
    lines = message.strip().split("\n")
    prefix     = f"{CY}{B}  Yami{R} │ "
    prefix_pad = f"       │ "
    for i, line in enumerate(lines):
        wrapped = textwrap.wrap(line, WIDTH - 10) or [""]
        for j, wline in enumerate(wrapped):
            if i == 0 and j == 0:
                print(f"{prefix}{wline}")
            else:
                print(f"{prefix_pad}{wline}")
    print()

def print_user(message: str):
    """Print user message with right-aligned label."""
    prefix     = f"{GR}{B}   You{R} │ "
    prefix_pad = f"       │ "
    lines = message.strip().split("\n")
    for i, line in enumerate(lines):
        if i == 0:
            print(f"{prefix}{line}")
        else:
            print(f"{prefix_pad}{line}")

def print_mcq(options: list):
    """Print MCQ options as numbered list."""
    print(f"       │")
    for i, opt in enumerate(options, 1):
        print(f"       │  {YL}[{i}]{R} {opt['label']}")
    print()

def print_status(msg: str, color=YL):
    """Print a subtle status line."""
    print(f"{color}{D}  ── {msg}{R}")

def print_section_change(section_name: str):
    """Print section transition banner."""
    print()
    print(f"{MG}{D}  ┌─ Section: {B}{section_name}{R}{MG}{D} {'─' * (WIDTH - 22 - len(section_name))}┐{R}")
    print()

def print_field_saves(diff: dict):
    """Print extracted fields as subtle green lines."""
    if not diff:
        return
    for key, val in diff.items():
        section, field = key.split(".", 1)
        print(f"{GR}{D}  ✓ {field.replace('_',' ').title()}: {val}{R}")
    if diff:
        print()

def print_command_hint():
    cmds = "/pause  /resume  /progress  /fields  /new  /quit  /help"
    print(f"{D}  Commands: {cmds}{R}")
    print()

def show_progress(session_id: str):
    try:
        data = get(f"/sessions/{session_id}/progress")
        print()
        hr("─", D)
        print(f"{B}  Progress{R}")
        for s in data["sections"]:
            icon = {"in_progress": "⏳", "complete": "✅", "pending": "🔜"}.get(s["status"], "·")
            pct  = s["completion_pct"]
            bar_filled = int(pct / 10)
            bar = "█" * bar_filled + "░" * (10 - bar_filled)
            print(f"  {icon} {s['display_name']:<25} {bar} {pct:>3}%")
        total = data["total_estimated_mins_remaining"]
        print()
        print(f"  {D}Total remaining: ~{total} min{R}")
        hr("─", D)
        print()
    except Exception as e:
        print(f"{RD}  Could not fetch progress: {e}{R}")

def show_fields(session_id: str):
    try:
        data = get(f"/sessions/{session_id}/fields")
        print()
        hr("─", D)
        print(f"{B}  Collected Fields{R}")
        if not data:
            print(f"  {D}(none yet){R}")
        for section, fields in data.items():
            filled = {k: v for k, v in fields.items() if v}
            if filled:
                print(f"  {MG}{section}{R}")
                for k, v in filled.items():
                    print(f"    {D}{k.replace('_',' ').title():<25}{R} {v}")
        hr("─", D)
        print()
    except Exception as e:
        print(f"{RD}  Could not fetch fields: {e}{R}")

def show_help():
    print()
    hr("─", D)
    cmds = [
        ("/up",       "Thumbs UP the last reply (optionally: /up it was clear)"),
        ("/down",     "Thumbs DOWN the last reply (optionally: /down asked twice)"),
        ("/alerts",   "Show open clinician alerts"),
        ("/pause",    "Pause the session"),
        ("/resume",   "Resume a paused session"),
        ("/progress", "Show section completion progress"),
        ("/fields",   "Show all extracted fields so far"),
        ("/new",      "Start a fresh session"),
        ("/quit",     "Exit the console"),
        ("/help",     "Show this message"),
    ]
    for cmd, desc in cmds:
        print(f"  {YL}{cmd:<12}{R} {desc}")
    hr("─", D)
    print()


# ── Field diff tracker ────────────────────────────────────────

def flatten_fields(data: dict) -> dict:
    """Flatten {section: {field: val}} → {section.field: val}"""
    flat = {}
    for section, fields in data.items():
        for k, v in fields.items():
            if v:
                flat[f"{section}.{k}"] = v
    return flat

def get_field_diff(session_id: str, prev_flat: dict) -> tuple[dict, dict]:
    """Return (new_fields_diff, updated_flat)."""
    try:
        data = get(f"/sessions/{session_id}/fields")
        current_flat = flatten_fields(data)
        diff = {k: v for k, v in current_flat.items() if prev_flat.get(k) != v}
        return diff, current_flat
    except Exception:
        return {}, prev_flat


# ── Session setup ─────────────────────────────────────────────

def setup_session() -> tuple[str, bool]:
    """
    Returns (session_id, is_fresh).
    Auto-generates UUID. Checks if session exists and asks user.
    """
    # Check if we have a saved session ID
    session_file = ".yami_session"
    existing_id = None

    if os.path.exists(session_file):
        with open(session_file) as f:
            existing_id = f.read().strip()

    if existing_id:
        try:
            session = get(f"/sessions/{existing_id}")
            status = session.get("status", "unknown")
            section = session.get("current_section", "unknown")
            print(f"{YL}  Found existing session: {D}{existing_id[:12]}...{R}")
            print(f"{YL}  Status: {status}  |  Section: {section}{R}")
            print()
            choice = input(f"  {B}Resume or start fresh?{R} [r/n]: ").strip().lower()
            if choice == "r":
                return existing_id, False
        except Exception:
            pass  # session not found on server, start fresh

    # Generate new UUID
    session_id = str(uuid.uuid4())[:12]
    with open(session_file, "w") as f:
        f.write(session_id)
    return session_id, True


# ── Main chat loop ────────────────────────────────────────────

def run():
    header()
    print_status("Connecting to Yami API...", CY)

    try:
        health = get("/health")
        if not health.get("model_loaded"):
            print_status("⚠  Model not loaded — responses will fail", RD)
        else:
            print_status("✓  Connected  |  Model ready", GR)
    except Exception as e:
        print(f"{RD}  ✗ Cannot connect to API at {BASE}: {e}{R}")
        print(f"{D}  Start server with: uvicorn api:app --port 5000{R}")
        sys.exit(1)

    print()

    session_id, is_fresh = setup_session()

    if is_fresh:
        print_status(f"New session: {session_id}", D)
        print()
        data = post("/sessions", {"user_id": session_id})
    else:
        print_status(f"Resuming session: {session_id}", D)
        # Re-fetch last bot message to orient user
        try:
            history = get(f"/sessions/{session_id}/history")
            messages = history.get("messages", [])
            last_bot = next(
                (m["content"] for m in reversed(messages) if m["role"] == "assistant"),
                None
            )
            if last_bot:
                print_bot(last_bot)
        except Exception:
            pass
        print_command_hint()
        flat_fields = flatten_fields(get(f"/sessions/{session_id}/fields"))
        current_section = get(f"/sessions/{session_id}").get("current_section", "")
        _chat_loop(session_id, current_section, flat_fields)
        return

    # Fresh session — show welcome + MCQ
    print_bot(data["bot_message"])
    if data.get("input_type") == "mcq":
        print_mcq(data["options"])

    print_command_hint()

    flat_fields = {}
    current_section = ""
    _chat_loop(session_id, current_section, flat_fields)


def _chat_loop(session_id: str, current_section: str, flat_fields: dict):
    """Main interactive loop."""
    options_map = {"1": "en", "2": "hi", "3": "hinglish", "4": "ta", "5":"pa", "6":"bn"}  # for MCQ shortcut
    waiting_for_mcq = current_section == ""  # fresh session starts in MCQ
    last_message_id = None                   # what /up and /down attach to

    while True:
        try:
            user_input = input(f"{GR}{B}   You{R} │ ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            print_status("Session saved. Goodbye 👋", CY)
            break

        if not user_input:
            continue

        # ── Commands ──
        if user_input.startswith("/"):
            cmd = user_input.lower().split()[0]
            arg = user_input[len(cmd):].strip()      # /down asked two things

            if cmd in ("/up", "/down"):
                # Feedback attaches to a message_id, so there has to be a reply
                # to attach it to. last_message_id is set on every bot turn.
                if not last_message_id:
                    print_status("Nothing to rate yet — send a message first.", YL)
                    continue
                try:
                    fb = post(f"/sessions/{session_id}/messages/{last_message_id}/feedback",
                              {"rating": cmd[1:], "review": arg or None})
                    mark = "👍" if cmd == "/up" else "👎"
                    note = f" — {fb['review']}" if fb.get("review") else ""
                    print_status(f"{mark} recorded as review #{fb['review_id']}{note}", GR)
                except Exception as e:
                    print_status(f"Could not record feedback: {e}", RD)
                continue

            elif cmd == "/alerts":
                try:
                    alerts = get("/admin/alerts")
                    if not alerts:
                        print_status("No open clinician alerts.", GR)
                    for a in alerts:
                        print(f"   {RD}{B}⚠ #{a['alert_id']} {a['risk']}{R} "
                              f"{D}{a['session_id']}{R}")
                        print(f"     {a['message'][:150]}")
                except Exception as e:
                    print_status(f"Could not fetch alerts: {e}", RD)
                continue

            elif cmd == "/quit":
                print_status("Session saved. Goodbye 👋", CY)
                break

            elif cmd == "/help":
                show_help()
                continue

            elif cmd == "/progress":
                show_progress(session_id)
                continue

            elif cmd == "/fields":
                show_fields(session_id)
                continue

            elif cmd == "/pause":
                try:
                    post(f"/sessions/{session_id}/pause")
                    print_status("💾 Session paused. Come back anytime.", YL)
                    print_status("Run the client again to resume.", D)
                    break
                except Exception as e:
                    print_status(f"Could not pause: {e}", RD)
                continue

            elif cmd == "/resume":
                try:
                    post(f"/sessions/{session_id}/resume")
                    print_status("✓ Session resumed.", GR)
                except Exception as e:
                    print_status(f"Could not resume: {e}", RD)
                continue

            elif cmd == "/new":
                confirm = input(f"  {RD}Start fresh? This keeps data but resets session. [y/n]: {R}").strip().lower()
                if confirm == "y":
                    data = post("/sessions", {"user_id": session_id})
                    print()
                    print_bot(data["bot_message"])
                    if data.get("input_type") == "mcq":
                        print_mcq(data["options"])
                    flat_fields = {}
                    current_section = ""
                    waiting_for_mcq = True
                continue

            else:
                print_status(f"Unknown command. Type /help for options.", D)
                continue

        # ── MCQ numeric shortcut ──
        if waiting_for_mcq and user_input in options_map:
            user_input = options_map[user_input]

        # ── Send message ──
        try:
            resp = post(
                f"/sessions/{session_id}/messages",
                {"message": user_input}
            )
        except requests.HTTPError as e:
            print_status(f"API error: {e}", RD)
            continue
        except Exception as e:
            print_status(f"Connection error: {e}", RD)
            continue

        # ── Section change detection ──
        new_section = resp.get("section", "")
        if new_section and new_section != current_section and current_section != "":
            print_section_change(new_section.replace("_", " ").title())
        current_section = new_section
        waiting_for_mcq = resp.get("input_type") == "mcq"

        # ── Bot response ──
        print()
        print_bot(resp["bot_message"])
        last_message_id = resp.get("message_id") or last_message_id
        if last_message_id:
            print(f"   {D}#{last_message_id} — /up or /down to rate this reply{R}")

        if resp.get("input_type") == "mcq" and resp.get("options"):
            print_mcq(resp["options"])

        # ── Session status changes ──
        risk = resp.get("risk")
        if risk:
            # Crisis triage is open. Show the level — "crisis" alone does not
            # distinguish passive ideation from imminent intent.
            print_status(f"⚠ CRISIS — risk: {risk.upper()}", RD)

        status = resp.get("session_status", "")
        if status == "crisis":
            print_status("Safety check in progress. The interview is on hold.", RD)
        elif status == "paused":
            print_status("💾 Session paused.", YL)
        elif status == "crisis":
            print_status("🆘 Crisis mode — please reach out for support.", RD)

        # ── Field extraction (brief wait, then show diff) ──
        if resp.get("input_type") == "text" and status not in ("paused", "crisis"):
            print(f"{D}  ⏳ extracting...{R}", end="", flush=True)
            time.sleep(4)
            diff, flat_fields = get_field_diff(session_id, flat_fields)
            clear_line()
            print_field_saves(diff)


# ── Entry point ───────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=BASE, help="API base URL")
    args = parser.parse_args()
    BASE = args.host

    run()

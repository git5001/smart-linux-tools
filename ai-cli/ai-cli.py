#!/usr/bin/env python3
"""
ai_shell_assistant.py  - Revamped CLI helper that converts natural-language
requests into shell commands (or concise answers) via OpenAI, **without** the
old in-program menu loop.

Key ideas
----------
* Stateless CLI sub-commands: ``ask``, ``chat``, ``add``, ``capture``, ``clear``.
* Captured output **only when explicitly wanted** (default for ``ask`` and
  ``capture``; use ``--silent`` to skip).
* Context and task persisted in a single JSON file under
  ``~/.cache/ai_shell_assistant/state.json``.
* Re-uses all proven helpers from the original script (env discovery, OpenAI
  calls, bash execution, formatting...).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import textwrap
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Union

from dotenv import load_dotenv  # type: ignore
from openai import OpenAI  # type: ignore
import platform
import re
from pathlib import Path as _P

from typing import TypedDict

# ---------------------------------------------------------------------------
# Configuration --------------------------------------------------------------
# ---------------------------------------------------------------------------

load_dotenv()
DEFAULT_MODEL = "gpt-4o-mini"
STRONG_MODEL = "gpt-4.1"
HISTORY_LINES = 10

# State file location (XDG-compliant)
STATE_FILE = Path(os.getenv("XDG_CACHE_HOME", "~/.cache"))
STATE_FILE = (STATE_FILE.expanduser() / "ai_shell_assistant" / "state.json")
STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise SystemExit("[fatal] Please export OPENAI_API_KEY or add it to .env")

client = OpenAI(api_key=API_KEY)

# ---------------------------------------------------------------------------
# Utility helpers copied / adapted from the original implementation
# ---------------------------------------------------------------------------



class ShellResult(TypedDict):
    stdout: str
    stderr: str
    returncode: int

def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def format_output(text: str, indent: str = "    ") -> str:
    width = shutil.get_terminal_size((76, 20)).columns
    wrapper = textwrap.TextWrapper(width=width, initial_indent="", subsequent_indent=indent)
    paragraphs = text.strip().split("\n")
    wrapped = [wrapper.fill(p) for p in paragraphs if p.strip()]
    return "\n".join(wrapped)


def env_info() -> Dict[str, str]:
    """Gather coarse environment info (MSYS2-aware)."""
    def _to_msys_path(win_path: str) -> str:
        p = win_path.replace("\\", "/")
        m = re.match(r"^([A-Za-z]):/(.*)", p)
        if m:
            drive, rest = m.groups()
            return f"/{drive.lower()}/{rest}"
        return p

    msys_env = os.getenv("MSYSTEM")
    if msys_env:
        raw_pwd, raw_home = os.getcwd().replace("\\", "/"), str(Path.home()).replace("\\", "/")
        pwd, home = _to_msys_path(raw_pwd), _to_msys_path(raw_home)
        raw_path = os.getenv("PATH", "")
        path = ":".join(_to_msys_path(e) for e in raw_path.split(os.pathsep) if e)
    else:
        pwd, home, path = os.getcwd(), str(Path.home()), os.getenv("PATH", "")

    u = platform.uname()
    now_dt = datetime.now(timezone.utc)
    now_iso = now_dt.isoformat(timespec="seconds").replace("+00:00", "Z")
    now_human = now_dt.strftime("%Y-%m-%d %H:%M:%S UTC")

    group = "unknown"
    try:
        import grp  # type: ignore
        group = grp.getgrgid(os.getgid()).gr_name
    except Exception:
        pass

    if msys_env:
        environment = f"MSYS2 ({msys_env})"
    else:
        try:
            with open("/etc/os-release") as f:
                environment = next(line for line in f if line.startswith("NAME=")).split("=", 1)[1].strip().strip('"')
        except Exception:
            environment = u.system

    return {
        "pwd": pwd,
        "PATH": path,
        "user": os.getenv("USER", os.getenv("USERNAME", "unknown")),
        "group": group,
        "home": home,
        "os": u.system,
        "uname": f"{u.system} {u.release} {u.machine}",
        "time_iso": now_iso,
        "time_human": now_human,
        "environment": environment,
        "shell": os.getenv("SHELL", os.getenv("COMSPEC", "unknown")),
    }


def tail_history(n: int = HISTORY_LINES) -> str:
    histfile = os.getenv("HISTFILE", str(Path.home() / ".bash_history"))
    try:
        return "\n".join(Path(histfile).read_text(encoding="utf-8", errors="ignore").splitlines()[-n:])
    except FileNotFoundError:
        try:
            out = subprocess.check_output(f"history | tail -n {n}", shell=True, text=True)
            return out.strip()
        except Exception:
            return ""


def _run_shell(cmd: str, capture: bool = False, debug: bool = False) -> ShellResult:
    if not cmd:
        return {"stdout": "", "stderr": "", "returncode": 0}
    bash_path = shutil.which("bash")
    if not bash_path:
        return {"stdout": "", "stderr": "[fatal] bash not found in PATH", "returncode": 127}
    full_cmd = [bash_path, "--login", "-i", "-c", cmd]
    if debug:
        print("Running bash cmd:", full_cmd)
    result = subprocess.run(full_cmd, text=True, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return {
        "stdout": result.stdout.strip() if result.stdout else "",
        "stderr": result.stderr.strip() if result.stderr else "",
        "returncode": result.returncode,
        }

# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

StateT = Dict[str, Union[str, List[Dict[str, object]]]]


def _default_state() -> StateT:
    return {
        "version": 1,
        "task": "",
        "mode": "command",
        "last_command": "",
        "history": [],
    }


def load_state() -> StateT:
    if STATE_FILE.exists():
        try:
            with STATE_FILE.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass  # corrupted -> start fresh
    return _default_state()


def save_state(state: StateT) -> None:
    with STATE_FILE.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def check_history_age(state: StateT, max_age_days: int = 2) -> None:
    """
    Prompt the user to clear history if the oldest entry is older than max_age_days.
    """
    hist = state.get("history", [])
    if not hist:
        return
    oldest = hist[0].get("timestamp")
    if not oldest:
        return
    try:
        ts = datetime.fromisoformat(oldest.replace("Z", "+00:00"))
    except ValueError:
        return
    if datetime.now(timezone.utc) - ts > timedelta(days=max_age_days):
        resp = input(f"Your history is over {max_age_days} days old. Clear it? [y/N] ")
        if resp.strip().lower() == "y":
            state["history"].clear()
            state.update({"task": "", "mode": "command", "last_command": ""})
            save_state(state)
            print("[ok] Old history cleared.")

# ---------------------------------------------------------------------------
# Prompt helpers (reuse original logic, but draw context from state["history"])
# ---------------------------------------------------------------------------


def _history_to_context(history: List[Dict[str, object]]) -> List[str]:
    ctx: List[str] = []
    skip_next = False

    for i, entry in enumerate(history):
        if skip_next:
            skip_next = False
            continue

        t = entry.get("type")

        if t == "note":
            ctx.append(f"Note: {entry['value']}")

        elif t == "command_generated":
            # Look ahead to see if it's immediately executed unchanged
            next_entry = history[i + 1] if i + 1 < len(history) else None
            if (
                next_entry and
                next_entry["type"] == "execution" and
                next_entry["value"]["command"].strip() == entry["value"].strip()
            ):
                cmd = entry["value"]
                out = next_entry["value"].get("output", "").strip()
                ctx.append(f"Command (AI-generated & executed): {cmd}\nOutput:\n{out}")
                skip_next = True
            else:
                ctx.append(f"Suggested command: {entry['value']}")

        elif t == "execution":
            v = entry["value"]
            ctx.append(f"Executed command: {v['command']}\nOutput:\n{v['output']}")

        elif t == "capture":
            v = entry["value"]
            ctx.append(f"Executed user command: {v['command']}\nOutput:\n{v['output']}")

        elif t == "chat":
            v = entry["value"]
            ctx.append(f"Q: {v['question']}\nA: {v['answer']}")

        elif t == "task":
            ctx.append(f"Task given: {entry['value']}")
        ctx.append("--------")     
    return ctx



def build_command_prompt(task: str, context: List[str], include_history: bool) -> str:
    pieces = [
        "You are an AI assistant that generates Linux commands based on a user task.",
        "Output *only* the command - no explanation, no backticks.",
        "---",
        *(f"{k}: {v}" for k, v in env_info().items()),
        "---",
        f"User query: {task}",
    ]
    if include_history:
        pieces.extend(["---", "Recent shell history:", tail_history()])
    if context:
        pieces.extend(["---", "Context:", *context])
    return "\n".join(pieces)


def build_chat_prompt(question: str, context: List[str], include_history: bool) -> str:
    intro = textwrap.dedent("""
        You are a technical CLI assistant that reasons about previous commands
        and their outputs. Use only the information in the context.
        - Never propose or output shell commands unless the user explicitly asks.
        - Answer user questions about the environment, past commands, and their results in plain text.
        - Be precise and succinct.
        - Make the output easy to read in a terminal (no fancy Unicode, proper line breaks).
        - Do not ask follow-up questions.
    """).strip()

    pieces = [intro, "---", *(f"{k}: {v}" for k, v in env_info().items())]
    if include_history:
        pieces.extend(["---", "Recent shell history:", tail_history()])
    if context:
        pieces.extend(["---", "Context:", *context])
    pieces.extend(["---", f"User question: {question}"])
    return "\n".join(pieces)


# ---------------------------------------------------------------------------
# OpenAI wrapper
# ---------------------------------------------------------------------------

def ask_openai(prompt: str, model: str) -> str:
    #return "pwd"
    resp = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}])
    return resp.choices[0].message.content.strip()

# ---------------------------------------------------------------------------
# CLI implementation
# ---------------------------------------------------------------------------

class Mode(Enum):
    COMMAND = auto()
    CHAT = auto()


def cmd_ask(args: argparse.Namespace, state: StateT) -> None:
    if args.again:
        if not state["task"]:
            print("[error] No previous task to repeat.")
            return
        task_text = state["task"]
    else:
        if not args.task:
            print("[error] Need a task description.")
            return
        task_text = " ".join(args.task)
        state["task"] = task_text
        state["mode"] = "command"
        state["history"].append({"type": "task", "value": task_text, "timestamp": _timestamp()})
    context_strings = _history_to_context(state["history"])
    prompt = build_command_prompt(task_text, context_strings, include_history=args.history)
    if args.debug:
        print("[debug prompt]\n" + prompt)
    command = ask_openai(prompt, STRONG_MODEL if args.strong else DEFAULT_MODEL)
    print(f"\n[command] {command}")
    state["last_command"] = command
    state["history"].append({"type": "command_generated", "value": command, "timestamp": _timestamp()})

    # Confirm execution unless --yes provided
    execute = args.yes or input("Execute [y/N]? ").strip().lower() == "y"
    if not execute:
        save_state(state)
        return

    silent = args.silent
    output = _run_shell(command, capture=not silent, debug=args.debug)
    if output["stdout"]:
        print(output["stdout"], end="\n")
    if output["stderr"]:
        print(output["stderr"], end="\n", file=sys.stderr)
    output_str = (
        f"Return code: {output['returncode']}\n"
        f"STDOUT:\n{output['stdout']}\n"
        f"STDERR:\n{output['stderr']}"
        )        
        
    if not silent:
        state["history"].append({
            "type": "execution",
            "value": {"command": command, "output": output_str, "returncode": output['returncode']},
            "timestamp": _timestamp(),
        })
    save_state(state)


def cmd_chat(args: argparse.Namespace, state: StateT) -> None:
    if args.again:
        # find last chat entry
        question = None
        for ent in reversed(state["history"]):
            if ent["type"] == "chat":
                question = ent["value"]["question"]
                break
        if not question:
            print("[error] No previous chat to repeat.")
            return
    else:
        if not args.question:
            print("[error] Need a question.")
            return
        question = " ".join(args.question)
    context_strings = _history_to_context(state["history"])
    prompt = build_chat_prompt(question, context_strings, include_history=args.history)
    if args.debug:
        print("[debug prompt]\n" + prompt)
    answer = ask_openai(prompt, STRONG_MODEL if args.strong else DEFAULT_MODEL)
    print("\n" + format_output(answer) + "\n")
    state["mode"] = "chat"
    state["history"].append({
        "type": "chat",
        "value": {"question": question, "answer": answer},
        "timestamp": _timestamp(),
    })
    save_state(state)


def cmd_add(args: argparse.Namespace, state: StateT) -> None:
    note = " ".join(args.note)
    if not note:
        print("[error] Need note text.")
        return
    state["history"].append({"type": "note", "value": note, "timestamp": _timestamp()})
    save_state(state)
    print("[ok] Note added to context.")


def cmd_capture(args: argparse.Namespace, state: StateT) -> None:
    cmd = " ".join(args.shell_cmd)
    if not cmd:
        print("[error] Need a shell command to capture.")
        return
    output = _run_shell(cmd, capture=True, debug=args.debug)
    if output["stdout"]:
        print(output["stdout"], end="\n")
    if output["stderr"]:
        print(output["stderr"], end="\n", file=sys.stderr)
    output_str = (
        f"Return code: {output['returncode']}\n"
        f"STDOUT:\n{output['stdout']}\n"
        f"STDERR:\n{output['stderr']}"
    )        
        
    state["history"].append({
        "type": "capture",
        "value": {"command": cmd, "output": output_str, "returncode": output['returncode']},
        "timestamp": _timestamp(),
    })
    save_state(state)


def cmd_context(ns: argparse.Namespace, st: StateT) -> None:
    """Show current task & history (or raw JSON)."""
    if ns.raw:
        print(json.dumps(st, indent=2))
        return

    print(f"Current task : {st['task'] or '(none)'}")
    print(f"Current mode : {st['mode']}")
    print(f"Last command : {st['last_command'] or '(none)'}")
    print("-" * 60)
    if not st["history"]:
        print("(context history is empty)")
        return
    for idx, e in enumerate(st["history"], 1):
        t = e['type']
        ts = e.get("timestamp", "?")
        print(f"{idx:02d}. [{t}] @ {ts}")
        if t in {"note", "task"}:
            print("    ", e["value"])
        elif t == "command_generated":
            print("    cmd:", e["value"])
        elif t in {"execution", "capture"}:
            print("    cmd :", e["value"]["command"])
            out = e["value"].get("output", "").strip()
            if out:
                # show first 10 non-blank lines to keep it readable
                lines = [ln for ln in out.splitlines() if ln.strip()][:10]
                hidden_count = len(out.splitlines()) - len(lines)
                more = f"... (+{hidden_count} more line{'s' if hidden_count != 1 else ''})" if hidden_count > 0 else ""
                print("    out :", "\n           ".join(lines) + "\n           " + more)
        elif t == "chat":
            print("    Q:", e["value"]["question"])
            print("    A:", e["value"]["answer"])
        print()
def cmd_clear(args: argparse.Namespace, state: StateT) -> None:
    """
    If args.count is None  ->  full reset (previous behaviour).
    If args.count is N     ->  drop the N oldest entries from state["history"].
    """

    # full reset ------------------------------------------------------------
    if args.count is None:
        if STATE_FILE.exists():
            STATE_FILE.unlink()
        print("[ok] Session cleared.")
        return

    # partial reset ---------------------------------------------------------
    n = args.count
    if n <= 0:
        print("[error] <number> must be a positive integer")
        return

    removed = min(n, len(state["history"]))
    state["history"] = state["history"][removed:]

    # If every entry is gone, also forget the "current task" header info
    if not state["history"]:
        state.update({"task": "", "mode": "command", "last_command": ""})

    save_state(state)
    print(f"[ok] Cleared {removed} entr{'y' if removed == 1 else 'ies'} "
          f"({len(state['history'])} remain).")

# ---------------------------------------------------------------------------
# Argument parsing setup
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ai_shell_assistant", description="Natural-language shell helper via OpenAI")
    sub = p.add_subparsers(dest="sub")

    # Common parent for shared flags
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("-d", "--debug", action="store_true", help="Debug output")
    parent.add_argument("-s", "--strong", action="store_true", help="Use stronger model")
    parent.add_argument("-H", "--history", action="store_true", help="Include last 10 shell-history lines in prompt")

    # ask
    ask_p = sub.add_parser("ask", parents=[parent], help="Generate (and optionally run) a shell command from a task")
    ask_p.add_argument("task", nargs="*", help="Task description (omit with --again)")
    ask_p.add_argument("-y", "--yes", action="store_true", help="Auto-confirm execution")
    ask_p.add_argument("--silent", action="store_true", help="Execute without storing output in context")
    ask_p.add_argument("--again", action="store_true", help="Repeat previous task")

    # chat
    chat_p = sub.add_parser("chat", parents=[parent], help="Ask a question about current context")
    chat_p.add_argument("question", nargs="*", help="Question text (omit with --again)")
    chat_p.add_argument("--again", action="store_true", help="Repeat previous chat")

    # add
    add_p = sub.add_parser("add", parents=[parent], help="Add a note to the context")
    add_p.add_argument("note", nargs=argparse.REMAINDER, help="Free-form note")

    # capture
    cap_p = sub.add_parser("capture", parents=[parent], help="Run a shell command and capture its output into context")
    cap_p.add_argument("shell_cmd", nargs=argparse.REMAINDER, help="Command to run")

    # context
    ctx = sub.add_parser("context", parents=[parent], help="Show stored context / task")
    ctx.add_argument("--raw", action="store_true", help="Dump raw JSON")

    # clear
    clr = sub.add_parser("clear", parents=[parent],
                         help="Clear the whole context or, if a number is given, "
                              "remove that many oldest history entries")
    clr.add_argument("count", nargs="?", type=int,
                     help="Number of history items to delete (oldest first)")
    return p

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    ns = parser.parse_args(argv)
    if ns.sub is None:
        parser.print_help()
        return
    state = load_state()

    # Prompt to clear old history
    check_history_age(state)

    if ns.sub == "ask":
        cmd_ask(ns, state)
    elif ns.sub == "chat":
        cmd_chat(ns, state)
    elif ns.sub == "add":
        cmd_add(ns, state)
    elif ns.sub == "capture":
        cmd_capture(ns, state)
    elif ns.sub == "context":
        cmd_context(ns, state)
    elif ns.sub == "clear":
        cmd_clear(ns, state)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

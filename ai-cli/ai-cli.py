#!/usr/bin/env python3
"""
ai_shell_assistant.py - A small, readable CLI helper that turns natural-language
requests into shell commands (or short answers) via OpenAI.

Highlights
----------
* **Minimal dependencies** - only `openai`, `python-dotenv`, and the standard library.
* **Single, explicit state-machine** for user interaction.
* Reads the API key from the environment (use a `.env` file if you like).
* `-s / --strong` flag switches to the larger model; `-h / --history` adds the
  last 10 shell-history lines to the prompt (privacy-off by default).
* Clear, short functions; generous docstrings.
"""

from __future__ import annotations

import os
import sys
import subprocess
import platform
from enum import Enum, auto
from pathlib import Path
from typing import List, Dict
from datetime import datetime, timezone

from dotenv import load_dotenv  # type: ignore
from openai import OpenAI  # type: ignore
import argparse
import textwrap

# Optional grp import for Unix-like systems
try:
    import grp
except ImportError:
    grp = None

# ---------------------------------------------------------------------------
# Configuration & helpers
# ---------------------------------------------------------------------------

load_dotenv()  # pull variables from .env into os.environ if present

DEFAULT_MODEL = "gpt-4o-mini"  # Cheap but reasonable model
STRONG_MODEL = "gpt-4.1"        # More capable (and expensive)
HISTORY_LINES = 10

# Validate API key early so we can fail fast and clearly
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    sys.exit("[fatal] Please export OPENAI_API_KEY or add it to your .env file.")

client = OpenAI(api_key=API_KEY)


def env_info() -> Dict[str, str]:
    """Return coarse information about the current environment, normalizing
    Windows drive paths into MSYS2 POSIX form when MSYSTEM is set."""
    import re

    def _to_msys_path(win_path: str) -> str:
        # unify slashes
        p = win_path.replace("\\", "/")
        # match drive-letter at the start
        m = re.match(r"^([A-Za-z]):/(.*)", p)
        if m:
            drive, rest = m.groups()
            return f"/{drive.lower()}/{rest}"
        return p

    # Detect MSYS2/UCRT
    msys_env = os.getenv("MSYSTEM")  # e.g. "UCRT64", "MINGW64", "MSYS"
    if msys_env:
        # normalize pwd and home
        raw_pwd  = os.getcwd().replace("\\", "/")
        raw_home = str(Path.home()).replace("\\", "/")
        pwd  = _to_msys_path(raw_pwd)
        home = _to_msys_path(raw_home)

        # normalize PATH entries
        raw_path = os.getenv("PATH", "")
        entries = [e for e in raw_path.split(os.pathsep) if e]
        norm_entries = [_to_msys_path(e) for e in entries]
        path = ":".join(norm_entries)
    else:
        # fallback for real Linux, macOS, plain Windows, etc.
        pwd  = os.getcwd()
        home = str(Path.home())
        path = os.getenv("PATH", "")

    # System uname info
    u = platform.uname()
    os_info = f"{u.system} {u.release} {u.machine}"

    # Current time in UTC
    now_dt = datetime.now(timezone.utc)
    now_iso = now_dt.isoformat(timespec='seconds').replace('+00:00', 'Z')  # ISO 8601
    now_human = now_dt.strftime("%Y-%m-%d %H:%M:%S UTC")  # Human-readable

    # User group (Unix only)
    if grp:
        try:
            group = grp.getgrgid(os.getgid()).gr_name
        except Exception:
            group = "unknown"
    else:
        group = "unknown"

    # Distro / environment label
    if msys_env:
        environment = f"MSYS2 ({msys_env})"
    else:
        try:
            with open("/etc/os-release") as f:
                environment = next(
                    line for line in f if line.startswith("NAME=")
                ).split("=", 1)[1].strip().strip('"')
        except Exception:
            environment = platform.system()

    return {
        "pwd": pwd,
        "PATH": path,
        "user": os.getenv("USER", os.getenv("USERNAME", "unknown")),
        "group": group,
        "home": home,
        "os": u.system,
        "uname": os_info,
        "time_iso": now_iso,
        "time_human": now_human,
        "environment": environment,
        "shell": os.getenv("SHELL", os.getenv("COMSPEC", "unknown")),
    }




def tail_history(n: int = HISTORY_LINES) -> str:
    """Return the last *n* lines from the interactive shell history (best-effort)."""
    histfile = os.getenv("HISTFILE", str(Path.home() / ".bash_history"))
    try:
        return "\n".join(Path(histfile).read_text(encoding="utf-8", errors="ignore").splitlines()[-n:])
    except FileNotFoundError:
        try:
            out = subprocess.check_output(f"history | tail -n {n}", shell=True, text=True)
            return out.strip()
        except Exception:
            return ""


def build_command_prompt(task: str, extra: List[str], include_history: bool) -> str:
    """Return a prompt that asks the model for a *shell command* only."""
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
    if extra:
        pieces.extend(["---", "Context:", *extra])
    return "\n".join(pieces)


def build_chat_prompt(query: str, extra: List[str], include_history: bool) -> str:
    """
    Return a prompt that asks the model for a *concise* answer about the
    existing context (commands run, their outputs, and the environment).
    The assistant should only emit actual shell commands if the user
    explicitly requests them.
    """
    intro = textwrap.dedent("""
        You are a technical CLI assistant that reasons about previous commands
        and their outputs. Use only the information in the context.
        o Never propose or output shell commands unless the user explicitly asks.
        o Answer user questions about the environment, past commands, and 
          their results in plain text.
        o Be precise and succinct: short answers when enough, longer when needed.
        o Do not ask follow-ups.
    """).strip()

    pieces = [
        intro,
        "---",
        # inject environment info
        *(f"{k}: {v}" for k, v in env_info().items()),
    ]

    if include_history:
        pieces.append("---")
        pieces.append("Recent shell history:")
        pieces.append(tail_history())

    if extra:
        pieces.append("---")
        pieces.append("Context:")
        pieces.extend(extra)

    pieces.extend([
        "---",
        f"User question: {query}",
    ])

    return "\n".join(pieces)


def ask_openai(prompt: str, model: str) -> str:
    """Return `message.content` from the first choice."""
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content.strip()


class Mode(Enum):
    COMMAND = auto()  # produce a shell command
    CHAT = auto()     # produce a short answer


def _run_shell(cmd: str, capture: bool = False, debug: bool = False) -> str:
    """Execute *cmd*; optionally capture and return its stdout+stderr."""
    if not cmd:
        return ""
    try:
        if capture:
            if debug:
                print("Running shell capture cmd ",cmd)
            res = subprocess.run(cmd, shell=True, text=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return res.stdout + res.stderr
        else:
            if debug:
                print("Running shell cmd ",cmd)
            subprocess.run(cmd, shell=True, check=True)
            return ""
    except subprocess.CalledProcessError as e:
        print(f"[error] {e}")
        return e.stdout + e.stderr if capture else ""

def build_menu(mode: Mode) -> str:
    return textwrap.dedent(
        f"""
        Options ({'CHAT' if mode == Mode.CHAT else 'COMMAND'}):
          q - quit
          e - execute command and quit
          x - execute and stay (captures output)
          s - silent execute (no capture output)
          u - USER: run *your* shell cmd & capture output (adds to context)
          t - talk (-> chat mode)
          a - add extra context line
          n - new CLI task (-> command mode)
          r - repeat LLM call
        """
    )

    
def session(task: str, model: str, include_history: bool, debug: bool, start_in_chat: bool = False) -> None:
    """Run the REPL-style loop."""
    mode = Mode.CHAT if start_in_chat else Mode.COMMAND
    extra: List[str] = []
    last_command = ""


    while True:
        if debug:
            print("Mode is ", mode)
        prompt = build_command_prompt(task, extra, include_history) if mode == Mode.COMMAND else build_chat_prompt(task, extra, include_history)
        if debug:
            print("[debug] LLM prompt:\n", prompt)
        answer = ask_openai(prompt, model)
        if mode == Mode.COMMAND:
            last_command = answer
            print(f"\n[command] {last_command}")
        else:
            print(f"\n[answer] {answer}")

        print(build_menu(mode))
        choice = input("choice > ").strip().lower()
        if choice == "q": break
        if choice in ("e", "x", "s", "u"):
            if choice == "e":
                output = _run_shell(last_command, capture=False, debug=debug)
                print(f"\n[output]\n{output.rstrip()}\n")
                break
            if choice == "s":
                output = _run_shell(last_command, capture=False, debug=debug)
                print(f"\n[output]\n{output.rstrip()}\n")
            if choice == "x":
                output = _run_shell(last_command, capture=True, debug=debug)
                print(f"\n[output]\n{output.rstrip()}\n")
                extra.append(f"Executed command: {last_command}\nOutput:\n{output}")
            if choice == "u":
                user_cmd = input("your cmd > ")
                output = _run_shell(user_cmd, capture=True, debug=debug)
                print(f"\n[output]\n{output.rstrip()}\n")
                extra.append(f"User command: {user_cmd}\nOutput:\n{output}")
        elif choice == "t":
            task = input("ask > ")
            mode = Mode.CHAT
        elif choice == "a":
            line = input("note > ")
            if line: extra.append(line)
        elif choice == "n":
            task = input("new task > ")
            mode = Mode.COMMAND
        elif choice == "r":
            continue
        else:
            print("!? unknown choice")


def main(argv: List[str] | None = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(
        add_help=False,
        prog=Path(__file__).name,
        description="Ask an LLM for shell commands or concise answers.",
    )
    parser.add_argument("task", nargs=argparse.REMAINDER, help="The user task / question")
    parser.add_argument("-d", "--debug", action="store_true", help="Show extra debug info")
    parser.add_argument("-s", "--strong", action="store_true", help="Use the stronger (more expensive) model")
    parser.add_argument("-h", "--history", action="store_true", help="Include the last 10 shell-history lines in the prompt")
    parser.add_argument("-?", "--help", action="help", help="Show this help and exit")
    parser.add_argument("-c", "--chat", action="store_true", help="Start directly in chat mode")


    ns = parser.parse_args(argv)
    if not ns.task:
        parser.error("Need a task - e.g. `ai_shell_assistant.py list all jpgs`.")

    task_text = " ".join(ns.task)
    model = STRONG_MODEL if ns.strong else DEFAULT_MODEL
    if ns.debug:
        print(f"[debug] model={model} history={ns.history}")

    session(task_text, model=model, include_history=ns.history, debug=ns.debug, start_in_chat=ns.chat)



if __name__ == "__main__":
    main()

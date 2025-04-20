#!/usr/bin/env python3
"""
ai_shell_assistant.py – A small, readable CLI helper that turns natural‑language
requests into shell commands (or short answers) via OpenAI.

Highlights
----------
* **Minimal dependencies** – only `openai`, `python-dotenv`, and the standard library.
* **Single, explicit state‑machine** for user interaction.
* Reads the API key from the environment (use a `.env` file if you like).
* `-s / --strong` flag switches to the larger model; `-h / --history` adds the
  last 10 shell‑history lines to the prompt (privacy‑off by default).
* Clear, short functions; generous docstrings.
"""

from __future__ import annotations

import os
import sys
import grp
import subprocess
from enum import Enum, auto
from pathlib import Path
from typing import List, Dict

from dotenv import load_dotenv  # type: ignore
from openai import OpenAI  # type: ignore
import argparse
import textwrap

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
    """Return coarse information about the current environment."""
    return {
        "pwd": os.getcwd(),
        "user": os.getenv("USER", "unknown"),
        "group": grp.getgrgid(os.getgid()).gr_name,
        "home": str(Path.home()),
        "os": os.uname().sysname,
        "shell": os.getenv("SHELL", "unknown"),
    }


def tail_history(n: int = HISTORY_LINES) -> str:
    """Return the last *n* lines from the interactive shell history (best‑effort)."""
    histfile = os.getenv("HISTFILE", str(Path.home() / ".bash_history"))
    try:
        return "\n".join(Path(histfile).read_text(encoding="utf‑8", errors="ignore").splitlines()[-n:])
    except FileNotFoundError:
        # Fallback: try calling `history` if we're inside an interactive shell.
        try:
            out = subprocess.check_output(f"history | tail -n {n}", shell=True, text=True)
            return out.strip()
        except Exception:
            return ""


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_command_prompt(task: str, extra: List[str], include_history: bool) -> str:
    """Return a prompt that asks the model for a *shell command* only."""
    pieces = [
        "You are an AI assistant that generates Linux commands based on a user task.",
        "Output *only* the command – no explanation, no backticks.",
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
    """Return a prompt that asks the model for a *concise* answer."""
    pieces = [
        "You are a short, no‑fluff cli assistant.",
        "Respond concisely (one or two sentences).",
        "---",
        *(f"{k}: {v}" for k, v in env_info().items()),
        "---",
        f"Direct user query: {query}",
    ]
    if include_history:
        pieces.extend(["---", "Recent shell history:", tail_history()])
    if extra:
        pieces.extend(["---", "Context:", *extra])
    return "\n".join(pieces)


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

def ask_openai(prompt: str, model: str) -> str:
    """Return `message.content` from the first choice."""
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

class Mode(Enum):
    COMMAND = auto()  # produce a shell command
    CHAT = auto()     # produce a short answer


# ---------------------------------------------------------------------------
# Interactive session
# ---------------------------------------------------------------------------

def session(task: str, model: str, include_history: bool, debug: bool) -> None:
    """Run the REPL‑style loop."""
    mode = Mode.COMMAND
    extra: List[str] = []          # accumulated context lines
    last_command = ""

    menu = textwrap.dedent(
        """
        Options:
          q – quit
          e – execute command and quit
          x – execute command and stay
          o – execute & capture output (adds to context)
          u – USER: run *your* shell cmd & capture output (adds to context)
          t – talk (chat mode)
          a – add extra context line
          n – new task
          r – repeat LLM call
        """
    )

    while True:
        if mode == Mode.COMMAND:
            prompt = build_command_prompt(task, extra, include_history)
        else:
            prompt = build_chat_prompt(task, extra, include_history)

        if debug:
            print("[debug] LLM prompt:\n", prompt)

        answer = ask_openai(prompt, model)

        if mode == Mode.COMMAND:
            last_command = answer
            print(f"\n[command] {last_command}")
        else:
            print(f"\n[answer] {answer}")

        print(menu)
        choice = input("choice › ").strip().lower()

        if choice == "q":
            break
        elif choice == "e":
            _run_shell(last_command, capture=False)
            break
        elif choice == "x":
            _run_shell(last_command, capture=False)
        elif choice == "o":
            output = _run_shell(last_command, capture=True)
            extra.append(f"Executed command: {last_command}\nOutput:\n{output}")
        elif choice == "u":
            user_cmd = input("your cmd › ")
            output = _run_shell(user_cmd, capture=True)
            extra.append(f"User command: {user_cmd}\nOutput:\n{output}")
        elif choice == "t":
            task = input("ask › ")
            mode = Mode.CHAT
        elif choice == "a":
            line = input("note › ")
            if line:
                extra.append(line)
        elif choice == "n":
            task = input("new task › ")
            mode = Mode.COMMAND
        elif choice == "r":
            continue  # loop triggers new API call immediately
        else:
            print("!? unknown choice")


# ---------------------------------------------------------------------------
# Shell execution helper
# ---------------------------------------------------------------------------

def _run_shell(cmd: str, capture: bool = False) -> str:
    """Execute *cmd*; optionally capture and return its stdout+stderr."""
    if not cmd:
        return ""

    try:
        if capture:
            res = subprocess.run(cmd, shell=True, text=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return res.stdout + res.stderr
        else:
            subprocess.run(cmd, shell=True, check=True)
            return ""
    except subprocess.CalledProcessError as e:
        print(f"[error] {e}")
        return e.stdout + e.stderr if capture else ""


# ---------------------------------------------------------------------------
# CLI entry‑point
# ---------------------------------------------------------------------------

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
    parser.add_argument("-h", "--history", action="store_true", help="Include the last 10 shell‑history lines in the prompt")
    parser.add_argument("-?", "--help", action="help", help="Show this help and exit")

    ns = parser.parse_args(argv)
    if not ns.task:
        parser.error("Need a task – e.g. `ai_shell_assistant.py list all jpgs`.")

    task_text = " ".join(ns.task)
    model = STRONG_MODEL if ns.strong else DEFAULT_MODEL

    if ns.debug:
        print(f"[debug] model={model} history={ns.history}")

    session(task_text, model=model, include_history=ns.history, debug=ns.debug)


if __name__ == "__main__":
    main()

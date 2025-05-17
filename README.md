# smart-linux-tools

**AI-powered CLI tools for smarter Linux workflows.**

A growing collection of lightweight, terminal-friendly tools that use OpenAI models to help you generate shell commands, capture command output, annotate your workflow, and answer questions — all from the command line.

---

## Included tools

### `ai_shell_assistant` – modular CLI with stateless subcommands

Natural-language Linux assistant with smart prompt construction, context-aware responses, and full history tracking.

#### Subcommands:
- `ask` — generate and (optionally) run a shell command from a prompt
- `chat` — ask contextual questions about previous commands
- `capture` — run a command and store its output in the assistant’s memory
- `add` — add a free-form note to the session history
- `context` — review current task and full interaction history
- `clear` — reset full or partial history

#### Example:
```bash
ai_shell_assistant ask -s list all .jpg files modified in the last 3 days
````

---

## Getting started

### 1. Set your OpenAI API key

Either export it:

```bash
export OPENAI_API_KEY=your-key-here
```

Or put it in a `.env` file:

```env
OPENAI_API_KEY=your-key-here
```

### 2. Install dependencies

Use a virtual environment if desired:

```bash
python3 -m venv ~/venv/cli-tools
source ~/venv/cli-tools/bin/activate
pip install python-dotenv openai
```

### 3. Make it easy to invoke

Create a shell alias or wrapper:

```bash
alias ask='ai_shell_assistant ask'
alias chat='ai_shell_assistant chat'
```

Or a script like `~/bin/ask`:

```bash
#!/bin/bash
python3 /path/to/ai_shell_assistant.py ask "$@"
```

Make it executable:

```bash
chmod +x ~/bin/ask
```

---

## Example workflows

```bash
ask install nginx and configure a basic site
chat why did the last command fail?
capture df -h
add Remember to clean /tmp weekly
context
clear 3
```

---

## Contributions welcome

Want to add new subcommands or models? Found a bug?
Open a PR or issue — let’s make Linux smarter together.


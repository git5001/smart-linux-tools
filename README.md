# 🧠 smart-linux-tools

**AI-powered CLI tools for smarter Linux workflows.**

This is a growing collection of lightweight, terminal-friendly tools that use language models like GPT to assist with common Linux command-line tasks — from generating shell commands to explaining output and helping you troubleshoot faster.

---

## 📦 Included tools

### `ai-cli` (`ask`) – natural language to shell command

Generate Linux commands from natural language prompts and run them interactively.

#### Features:
- 🔁 REPL-style loop with options to run, inspect, or modify commands
- 💬 Chat mode: ask follow-up questions or get explanations
- 🔐 Optional history injection (privacy-respecting, opt-in)
- 🧠 Smart prompt construction using environment data
- 🧬 Supports both lightweight and strong OpenAI models (`gpt-4.1`, `gpt-4o-mini`)

#### Example:
```bash
ask -s -h list all jpg files modified in the last 3 days
```

---

## 🚀 Getting started

### 1. Set your OpenAI API key
Either export it:
```bash
export OPENAI_API_KEY=your-key-here
```

Or put it in a `.env` file in the project directory:
```env
OPENAI_API_KEY=your-key-here
```

### 2. Create a virtual environment and install dependencies
```bash
python3 -m venv ~/venv/cli-tools/
source ~/venv/cli-tools/bin/activate
pip install python-dotenv openai
```

### 3. Create a shell wrapper for convenience
Save this as `ask` somewhere in your `PATH` (e.g. `~/bin/ask`):

```bash
#!/bin/bash
~/venv/cli-tools/bin/python ~/bin/ask.py \"$@\"
```
Make it executable:
```bash
chmod +x ~/bin/ask
```

### 4. Run the tool
```bash
ask list files in this folder sorted by size
```

---

## 🤝 Contributions welcome!

Have an idea for another smart CLI tool? Want to extend the assistant?  
Feel free to open issues or pull requests. Let’s make Linux a little smarter together.
```


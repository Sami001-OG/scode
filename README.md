# scode

[![GitHub release](https://img.shields.io/github/release/Sami001-OG/scode.svg)](https://github.com/Sami001-OG/scode/releases)
[![GitHub issues](https://img.shields.io/github/issues/Sami001-OG/scode.svg)](https://github.com/Sami001-OG/scode/issues)
[![GitHub license](https://img.shields.io/github/license/Sami001-OG/scode.svg)](https://github.com/Sami001-OG/scode/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)

A comprehensive CLI AI coding harness in a single Python file — designed for speed, accuracy, and low token usage. Works everywhere: Linux, macOS, Windows, Android (Termux/Pydroid). Zero native dependencies; only `httpx` and `rich` required.

## ✨ Features

- **Multi‑provider**: NVIDIA, OpenAI, Anthropic, Ollama, OpenRouter, Groq, LM Studio, and more — switch with `/provider` and `/model`.
- **Harness abstraction**: Run as native, or emulate popular agents (`claude-code`, `kimi`, `qwen`, `deepseek`, `swe-agent`, `zcode`, `minimal`) via `/harness`.
- **Rich toolset**: file I/O, shell commands, grep/rg/find, todo, web search/fetch, git, MCP servers, skill system, Agent‑to‑Agent Protocol (ACP).
- **Session persistence**: automatic save/load of chat history, config, and working directory.
- **Plan mode & steering**: `/plan` toggle, `Enter` to steer, `Alt+Enter` to queue follow‑ups.
- **Subagents & orchestrator**: delegate tasks to isolated workers with `delegate_task`.
- **Cross‑platform TUI**: beautiful, responsive interface powered by Rich.
- **No native deps**: pure‑Python, works on constrained devices (tested on 3.8 GB RAM, no GPU).
- **Secure**: never logs or exposes API keys/tokens; uses environment or encrypted config.

## 🚀 Installation

### Option 1: Git (recommended for contributors)
```bash
git clone https://github.com/Sami001-OG/scode.git
cd scode
./scode.py          # or: python3 scode.py
```

### Option 2: Quick install (one‑liner)
```bash
mkdir -p ~/.local/bin && \
curl -L https://raw.githubusercontent.com/Sami001-OG/scode/main/scode.py -o ~/.local/bin/scode && \
chmod +x ~/.local/bin/scode && \
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.profile && \
source ~/.profile   # or open a new terminal
```
After this, `scode` works from **any directory**.

### Option 3: Pip (if you prefer)
```bash
pip install git+https://github.com/Sami001-OG/scode.git
scode
```

## 📖 Usage

Launch the harness:
```bash
scode
```
You’ll see the TUI with a banner and a prompt. Available slash commands:

| Command | Description |
|---------|-------------|
| `/help` | Show all commands and keybindings |
| `/model` | Switch the AI model (e.g., `nemotron-3-nano`) |
| `/provider` | Switch provider (NVIDIA, OpenAI, Ollama, …) |
| `/harness` | Change runtime harness (`native`, `claude-code`, …) |
| `/plan` | Toggle plan mode (shows reasoning steps) |
| `/shell` | Toggle auto‑approve for shell commands |
| `/write` | Toggle auto‑approve for file writes |
| `/mcp` | Manage Model Context Protocol servers |
| `/skill` | Load, list, or create skills |
| `/acp` | Start ACP server for editor integration (VSCodium, Cursor, etc.) |
| `/session` | Save, load, or clear sessions |
| `/clear` | Erase current conversation |
| `/history` | Show recent session history |
| `/quit` | Exit the harness |

During a run:
- **Enter**: steer/redirect the agent based on latest output
- **Alt+Enter**: queue a follow‑up instruction to run after current task
- **Ctrl+C**: interrupt the current operation

### Example
```bash
scode "Create a Python script that prints 'Hello, World!' and save it as hello.py"
```
The agent will write the file, run it, and show the output.

## 📄 License

This project is licensed under the **Apache License 2.0** – see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

Built with ❤️ using [Hermes Agent](https://hermes-agent.nousresearch.com) and the amazing open‑source AI community.

Happy coding! 🚀
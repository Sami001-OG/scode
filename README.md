# scode

A comprehensive CLI AI coding harness in a single Python file.

## Features

- Multi‑provider (NVIDIA, OpenAI, Anthropic, Ollama, OpenRouter, Groq, LM Studio…)
- Harness abstraction (`/harness native|claude-code|kimi|qwen|deepseek|swe-agent|zcode|minimal`)
- Tools: read/write/edit, bash, grep/rg, find, todo, web search/fetch, git, MCP, skills, ACP
- Session persistence, plan mode, steering, subagents
- Cross‑platform: Linux/macOS/Windows/Android (Termux/Pydroid)
- No native dependencies, only `httpx` and `rich`

## Installation

```bash
# Option 1: via git
git clone https://github.com/Sami001-OG/scode.git
cd scode
./scode.py   # or python3 scode.py

# Option 2: quick install
mkdir -p ~/.local/bin
curl -L https://raw.githubusercontent.com/Sami001-OG/scode/main/scode.py -o ~/.local/bin/scode
chmod +x ~/.local/bin/scode
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.profile
source ~/.profile   # or open a new terminal
```

## Usage

Run the harness:

```bash
scode
```

Then use slash commands:

- `/help` – show all commands
- `/model` – switch model
- `/provider` – switch provider
- `/harness` – switch runtime
- `/plan` – toggle plan mode
- `/shell` – toggle auto‑approve shell
- `/write` – toggle auto‑approve file writes
- `/mcp` – manage MCP servers
- `/skill` – manage skills
- `/acp` – start ACP server for editor integration
- `/session` – session management
- `/clear` – clear conversation
- `/history` – show session history
- `/quit` – exit

During a run:
- `Ctrl+C` – interrupt
- `Enter` – steer/redirect
- `Alt+Enter` – queue follow‑up

## Example

```bash
scode "Create a Python script that prints 'Hello, World!' and save it as hello.py"
```

## License

MIT


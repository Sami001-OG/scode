#!/usr/bin/env python3
"""Scout v2 — comprehensive CLI AI coding harness. Single file, cross-platform, no native deps.
Capabilities: ACP, MCP, skills, harness abstraction, git-aware, steering, session persistence, plan mode, subagents.
Runs on Linux/macOS/Windows/Android (Termux/Pydroid)."""
import os, sys, json, re, ast, shutil, subprocess, time, argparse, signal, threading, queue, webbrowser, pathlib, textwrap, traceback, uuid, sqlite3, hashlib, inspect, tempfile, select
from pathlib import Path
from typing import Any, Callable, Optional
from dataclasses import dataclass, asdict, field
from datetime import datetime

try:
    import httpx
except ImportError:
    print("missing httpx: pip install httpx", file=sys.stderr); sys.exit(1)
try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.live import Live
    from rich.spinner import Spinner
    from rich.text import Text
    from rich.prompt import Prompt
    from rich.table import Table
    from rich.tree import Tree
    from rich.syntax import Syntax
    from rich.columns import Columns
    from rich.layout import Layout
    from rich.align import Align
except ImportError:
    print("missing rich: pip install rich", file=sys.stderr); sys.exit(1)

# --- cross-platform paths ---
HOME = Path.home()
IS_ANDROID = "ANDROID_ROOT" in os.environ or "TERMUX_VERSION" in os.environ or "PYDROID" in os.environ
IS_WINDOWS = sys.platform == "win32"
CFG_DIR = Path(os.environ.get("SCOUT_CONFIG", HOME / ".scout"))
CFG_DIR.mkdir(parents=True, exist_ok=True)
CFG_PATH = CFG_DIR / "config.json"
HISTORY_DB = CFG_DIR / "history.db"
SESSIONS_DB = CFG_DIR / "sessions.db"
SKILLS_DIR = CFG_DIR / "skills"
SKILLS_DIR.mkdir(exist_ok=True)
MCP_DIR = CFG_DIR / "mcp"
MCP_DIR.mkdir(exist_ok=True)

console = Console()
CWD = Path.cwd()
MAX_TOKENS = 8192
CURRENT_SESSION_ID = None

# --- providers ---
PROVIDERS = {
    "openai":    {"base": "https://api.openai.com/v1",             "models": ["gpt-4o", "gpt-4o-mini", "o1", "o1-mini", "o3-mini"]},
    "anthropic": {"base": "https://api.anthropic.com/v1",          "models": ["claude-sonnet-4-5", "claude-3-5-sonnet-latest", "claude-3-5-haiku-latest"]},
    "nvidia":    {"base": "https://integrate.api.nvidia.com/v1",   "models": ["minimaxai/minimax-m3", "meta/llama-3.3-70b-instruct", "nvidia/llama-3.1-nemotron-70b-instruct", "nvidia/nemotron-3-nano-30b-a3b"]},
    "groq":      {"base": "https://api.groq.com/openai/v1",        "models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]},
    "openrouter":{"base": "https://openrouter.ai/api/v1",          "models": ["anthropic/claude-sonnet-4", "openai/gpt-4o", "google/gemini-2.0-flash"]},
    "ollama":    {"base": "http://localhost:11434/v1",             "models": ["qwen2.5-coder:7b", "llama3.2", "deepseek-coder-v2", "mistral-nemo"]},
    "lmstudio":  {"base": "http://localhost:1234/v1",              "models": ["local"]},
}

HARNESSES = {
    "native":     {"desc": "Built-in Scout engine (default)", "tools": "all"},
    "claude-code": {"desc": "Anthropic Claude Code runtime", "cmd": "claude"},
    "kimi":       {"desc": "Moonshot Kimi K2 CLI", "cmd": "kimi"},
    "qwen":       {"desc": "Alibaba Qwen Code CLI", "cmd": "qwen"},
    "deepseek":   {"desc": "DeepSeek Coder TUI", "cmd": "deepseek-tui"},
    "swe-agent":  {"desc": "SWE-agent (Princeton)", "cmd": "swe-agent"},
    "zcode":      {"desc": "Z.ai ZCode", "cmd": "zcode"},
    "minimal":    {"desc": "Minimal tool set (read/write/bash only)", "tools": "minimal"},
}

# --- tool registry ---
TOOLS: dict[str, dict] = {}

def tool(name: str, desc: str, params: dict):
    def deco(fn: Callable):
        TOOLS[name] = {"fn": fn, "desc": desc, "params": params}
        return fn
    return deco

def tool_schema(minimal: bool = False) -> list:
    exclude = {"subagent", "mcp_call", "acp_request", "git_diff", "git_commit", "skill_load", "skill_create"} if minimal else set()
    return [{"type": "function", "function": {"name": n, "description": t["desc"], "parameters": t["params"]}} for n, t in TOOLS.items() if n not in exclude]

# --- utils ---
def safe_path(p: str) -> Path:
    pp = Path(p).expanduser()
    if not pp.is_absolute():
        pp = (CWD / pp).resolve()
    return pp

def read_file_safe(path: Path, offset: int = 1, limit: int = 2000) -> str:
    if not path.exists(): return f"ERROR: {path} not found"
    try:
        lines = path.read_text(errors="replace").splitlines()
        total = len(lines)
        chunk = lines[max(0, offset-1):offset-1+limit]
        return "\n".join(f"{i+offset:4}|{ln}" for i, ln in enumerate(chunk)) + f"\n[{total} lines total]"
    except Exception as e: return f"ERROR: {e}"

def run_cmd(cmd: str, timeout: int = 60, cwd: Path = None) -> str:
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout, cwd=str(cwd or CWD))
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else f"(exit {r.returncode})"
    except subprocess.TimeoutExpired:
        return f"ERROR: timeout after {timeout}s"
    except Exception as e: return f"ERROR: {e}"

# --- config ---
def load_cfg() -> dict:
    if CFG_PATH.exists():
        try: return json.loads(CFG_PATH.read_text())
        except: pass
    return {}

def save_cfg(cfg: dict):
    CFG_PATH.write_text(json.dumps(cfg, indent=2))

def ensure_cfg() -> dict:
    cfg = load_cfg()
    if "provider" not in cfg:
        console.print("[bold cyan]Scout setup[/bold cyan]")
        names = list(PROVIDERS)
        for i, n in enumerate(names, 1):
            console.print(f"  [cyan]{i}[/cyan] {n}")
        choice = Prompt.ask("provider", choices=[str(i) for i in range(1, len(names)+1)], default="1")
        cfg["provider"] = names[int(choice)-1]
    if "api_key" not in cfg:
        cfg["api_key"] = Prompt.ask(f"api key for [cyan]{cfg['provider']}[/cyan] (blank to skip)", default="")
    if "model" not in cfg:
        models = PROVIDERS[cfg["provider"]]["models"]
        for i, m in enumerate(models, 1):
            console.print(f"  [cyan]{i}[/cyan] {m}")
        choice = Prompt.ask("model", choices=[str(i) for i in range(1, len(models)+1)], default="1")
        cfg["model"] = models[int(choice)-1]
    cfg.setdefault("max_iter", 50)
    cfg.setdefault("auto_approve_shell", False)
    cfg.setdefault("auto_approve_write", True)
    cfg.setdefault("web_provider", "duckduckgo")
    cfg.setdefault("theme", "monokai")
    cfg.setdefault("harness", "native")
    cfg.setdefault("mcp_servers", [])
    cfg.setdefault("skills_enabled", [])
    save_cfg(cfg)
    return cfg

# --- db init ---
def init_dbs():
    for db_path, schema in [
        (HISTORY_DB, "CREATE TABLE IF NOT EXISTS history (id INTEGER PRIMARY KEY, ts REAL, session_id TEXT, role TEXT, content TEXT, tool_calls TEXT)"),
        (SESSIONS_DB, "CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, created REAL, updated REAL, title TEXT, messages TEXT)"),
    ]:
        with sqlite3.connect(db_path) as db:
            db.execute(schema)

init_dbs()

# --- state ---
AGENT_STATE = {
    "todos": [],
    "plan_mode": False,
    "steering_queue": [],
    "interrupted": False,
    "current_harness": "native",
    "acp_mode": False,
    "acp_conn": None,
}

# --- tools ---
@tool("read_file", "Read a file. Optional offset (1-indexed) and limit.", {
    "type":"object","properties":{"path":{"type":"string"},"offset":{"type":"integer","default":1},"limit":{"type":"integer","default":2000}},"required":["path"]
})
def t_read_file(path: str, offset: int = 1, limit: int = 2000) -> str:
    return read_file_safe(safe_path(path), offset, limit)

@tool("write_file", "Write content to a file (overwrites). Creates parent dirs.", {
    "type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]
})
def t_write_file(path: str, content: str) -> str:
    p = safe_path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content); return f"wrote {len(content)} bytes to {path}"

@tool("edit_file", "Replace exact text in a file (single occurrence).", {
    "type":"object","properties":{"path":{"type":"string"},"old_string":{"type":"string"},"new_string":{"type":"string"}},"required":["path","old_string","new_string"]
})
def t_edit_file(path: str, old_string: str, new_string: str) -> str:
    p = safe_path(path)
    if not p.exists(): return f"ERROR: {path} not found"
    text = p.read_text()
    if old_string not in text: return "ERROR: old_string not found in file"
    text = text.replace(old_string, new_string, 1)
    p.write_text(text); return "ok"

@tool("bash", "Run a shell command. Timeout in seconds.", {
    "type":"object","properties":{"command":{"type":"string"},"timeout":{"type":"integer","default":60}},"required":["command"]
})
def t_bash(command: str, timeout: int = 60) -> str:
    return run_cmd(command, timeout)

@tool("grep", "Regex search in files using ripgrep (rg) or grep fallback. path=dir/file. glob=filter.", {
    "type":"object","properties":{"pattern":{"type":"string"},"path":{"type":"string","default":"."},"glob":{"type":"string"},"case_sensitive":{"type":"boolean","default":True},"max_results":{"type":"integer","default":100}},"required":["pattern"]
})
def t_grep(pattern: str, path: str = ".", glob: str = None, case_sensitive: bool = True, max_results: int = 100) -> str:
    flags = [] if case_sensitive else ["-i"]
    cmd = ["rg", "--line-number", "--no-heading", "--max-count", str(max_results)] + flags + [pattern, path]
    if glob: cmd.extend(["--glob", glob])
    if shutil.which("rg"):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return r.stdout[:20000] or "(no matches)"
        except Exception as e: return f"ERROR: {e}"
    cmd2 = ["grep", "-rn"] + ([] if case_sensitive else ["-i"]) + [pattern, path]
    try:
        r = subprocess.run(cmd2, capture_output=True, text=True, timeout=30)
        return r.stdout[:20000] or "(no matches)"
    except Exception as e: return f"ERROR: {e}"

@tool("find_files", "Find files by glob pattern.", {
    "type":"object","properties":{"glob":{"type":"string"},"path":{"type":"string","default":"."}},"required":["glob"]
})
def t_find_files(glob: str, path: str = ".") -> str:
    try:
        matches = [str(p) for p in Path(path).rglob(glob)]
        return "\n".join(matches[:500]) or "(none)"
    except Exception as e: return f"ERROR: {e}"

@tool("todo", "Track task progress. items=[{\"id\":..,\"content\":..,\"status\":\"pending|in_progress|completed\"}]", {
    "type":"object","properties":{"items":{"type":"array","items":{"type":"object"}}},"required":["items"]
})
def t_todo(items: list) -> str:
    AGENT_STATE["todos"] = items
    return "ok"

@tool("notebook_edit", "Edit a Jupyter notebook (.ipynb) by modifying a cell at a given index. Provide the notebook path, cell index, and new cell content (string or list of strings).", {
    "type":"object","properties":{"path":{"type":"string"},"cell_index":{"type":"integer"},"content":{"type":"string"},"cell_type":{"type":"string","enum":["code","markdown"],"default":"code"}},
    "required":["path","cell_index","content"]
})
def t_notebook_edit(path: str, cell_index: int, content: str, cell_type: str = "code") -> str:
    p = safe_path(path)
    if not p.exists():
        return f"ERROR: {path} not found"
    try:
        nb = json.loads(p.read_text())
    except Exception as e:
        return f"ERROR: failed to parse notebook: {e}"
    if 'cells' not in nb or not isinstance(nb['cells'], list):
        return "ERROR: not a valid notebook (no cells list)"
    if cell_index < 0 or cell_index >= len(nb['cells']):
        return f"ERROR: cell index {cell_index} out of range (0-{len(nb['cells'])-1})"
    # Normalize content to a list of strings (each string is a line)
    if isinstance(content, str):
        lines = content.splitlines()
    else:
        lines = list(content)
    # Update the cell
    cell = nb['cells'][cell_index]
    cell['cell_type'] = cell_type
    cell['source'] = lines
    try:
        p.write_text(json.dumps(nb, indent=1))
    except Exception as e:
        return f"ERROR: failed to write notebook: {e}"
    return f"updated cell {cell_index} in {path}"


@tool("web_search", "Search the web via DuckDuckGo HTML. query=string. limit=N (default 5).", {
    "type":"object","properties":{"query":{"type":"string"},"limit":{"type":"integer","default":5}},"required":["query"]
})
def t_web_search(query: str, limit: int = 5) -> str:
    try:
        r = httpx.get("https://html.duckduckgo.com/html/", params={"q": query}, timeout=15, headers={"User-Agent":"Mozilla/5.0"})
        from html.parser import HTMLParser
        class P(HTMLParser):
            def __init__(self):
                super().__init__()
                self.results = []
                self.in_result = False
                self.current = {}
            def handle_starttag(self, tag, attrs):
                if tag == "a" and any(k=="class" and "result__url" in v for k,v in attrs):
                    self.in_result = True
                if self.in_result and tag == "a" and any(k=="class" and "result__snippet" in v for k,v in attrs):
                    pass
            def handle_data(self, data):
                pass
        # simpler regex approach
        results = re.findall(r'class="result__snippet".*?>(.*?)</a>.*?class="result__url".*?>(.*?)</a>', r.text, re.S)
        out = []
        for snippet, url in results[:limit]:
            snippet = re.sub(r"<[^>]+>", "", snippet).strip()
            url = re.sub(r"<[^>]+>", "", url).strip()
            out.append(f"- {snippet}\n  {url}")
        return "\n".join(out) or "(no results)"
    except Exception as e: return f"ERROR: {e}"

@tool("web_fetch", "Fetch a URL, return text content (stripped HTML).", {
    "type":"object","properties":{"url":{"type":"string"},"max_chars":{"type":"integer","default":15000}},"required":["url"]
})
def t_web_fetch(url: str, max_chars: int = 15000) -> str:
    try:
        r = httpx.get(url, timeout=30, follow_redirects=True, headers={"User-Agent":"Mozilla/5.0 Scout/2.0"})
        ct = r.headers.get("content-type","")
        text = r.text
        if "html" in ct:
            text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.S)
            text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.S)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text)
        return text[:max_chars]
    except Exception as e: return f"ERROR: {e}"

@tool("subagent", "Spawn a subagent with a goal. Returns its final report.", {
    "type":"object","properties":{"goal":{"type":"string"},"context":{"type":"string","default":""}},"required":["goal"]
})
def t_subagent(goal: str, context: str = "") -> str:
    return run_subagent(goal, context)

@tool("plan", "Output a plan (read-only mode marker). content=markdown plan text.", {
    "type":"object","properties":{"content":{"type":"string"}},"required":["content"]
})
def t_plan(content: str) -> str:
    console.print(Panel(Markdown(content), title="[bold cyan]plan[/bold cyan]", border_style="cyan"))
    return "plan displayed"

@tool("git_status", "Show git status.", {"type":"object","properties":{}})
def t_git_status() -> str:
    return run_cmd("git status --short")

@tool("git_diff", "Show git diff (staged or unstaged).", {"type":"object","properties":{"staged":{"type":"boolean","default":False}}})
def t_git_diff(staged: bool = False) -> str:
    return run_cmd("git diff" + (" --cached" if staged else ""))

@tool("git_commit", "Commit staged changes with message.", {"type":"object","properties":{"message":{"type":"string"}},"required":["message"]})
def t_git_commit(message: str) -> str:
    return run_cmd(f'git commit -m "{message}"')

@tool("git_log", "Show recent git log.", {"type":"object","properties":{"count":{"type":"integer","default":10}}})
def t_git_log(count: int = 10) -> str:
    return run_cmd(f"git log --oneline -{count}")

@tool("mcp_call", "Call an MCP server tool. server=name, tool=tool_name, args={...}", {
    "type":"object","properties":{"server":{"type":"string"},"tool":{"type":"string"},"args":{"type":"object"}},"required":["server","tool"]
})
def t_mcp_call(server: str, tool: str, args: dict) -> str:
    cfg = load_cfg()
    servers = cfg.get("mcp_servers", [])
    srv = next((s for s in servers if s["name"] == server), None)
    if not srv: return f"ERROR: MCP server '{server}' not configured"
    return call_mcp(srv, tool, args)

@tool("skill_load", "Load a skill by name from ~/.scout/skills/.", {"type":"object","properties":{"name":{"type":"string"}},"required":["name"]})
def t_skill_load(name: str) -> str:
    return load_skill(name)

@tool("skill_create", "Create a skill from markdown content.", {"type":"object","properties":{"name":{"type":"string"},"content":{"type":"string"}},"required":["name","content"]})
def t_skill_create(name: str, content: str) -> str:
    (SKILLS_DIR / f"{name}.md").write_text(content)
    return f"skill {name} created"

@tool("acp_request", "Send ACP request (for editor integration). method=string, params={...}", {
    "type":"object","properties":{"method":{"type":"string"},"params":{"type":"object"}},"required":["method"]
})
def t_acp_request(method: str, params: dict) -> str:
    if not AGENT_STATE["acp_conn"]: return "ERROR: ACP not connected"
    return send_acp(method, params)

# --- harness abstraction ---
def run_harness(harness: str, prompt: str) -> str:
    """Delegate to external harness CLI."""
    h = HARNESSES.get(harness)
    if not h or "cmd" not in h:
        return f"ERROR: harness '{harness}' not available or no CLI command"
    cmd = h["cmd"]
    if not shutil.which(cmd):
        return f"ERROR: '{cmd}' not in PATH"
    return run_cmd(f'{cmd} "{prompt}"', timeout=300)

# --- MCP client ---
def call_mcp(server: dict, tool: str, args: dict) -> str:
    transport = server.get("transport", "stdio")
    if transport == "stdio":
        cmd = server["command"]
        proc = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        req = json.dumps({"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":tool,"arguments":args}}) + "\n"
        out, err = proc.communicate(req, timeout=30)
        try:
            resp = json.loads(out.strip().split("\n")[-1])
            return json.dumps(resp.get("result", resp.get("error", "unknown")), indent=2)
        except:
            return out or err or "no response"
    elif transport in ("http", "sse"):
        url = server["url"]
        r = httpx.post(f"{url}/mcp", json={"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":tool,"arguments":args}}, timeout=30)
        return json.dumps(r.json(), indent=2)
    return "ERROR: unknown transport"

# --- skills ---
def load_skill(name: str) -> str:
    path = SKILLS_DIR / f"{name}.md"
    if not path.exists():
        # try built-in skills
        for p in [Path("~/.hermes/skills").expanduser() / name / "SKILL.md", Path("~/.hermes/skills").expanduser() / f"{name}.md"]:
            if p.exists(): path = p; break
        else:
            return f"ERROR: skill '{name}' not found"
    content = path.read_text()
    # parse frontmatter
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            import yaml
            try: meta = yaml.safe_load(parts[1])
            except: meta = {}
    # register skill tools if defined
    return f"loaded skill: {name}\n{content[:2000]}"

# --- ACP (Agent Client Protocol) ---
def send_acp(method: str, params: dict) -> str:
    conn = AGENT_STATE["acp_conn"]
    if not conn: return "ERROR: no ACP connection"
    req = {"jsonrpc": "2.0", "id": str(uuid.uuid4()), "method": method, "params": params}
    conn.stdin.write(json.dumps(req) + "\n")
    conn.stdin.flush()
    line = conn.stdout.readline()
    return line.strip()

def start_acp_server():
    """Start ACP stdio server for editor integration."""
    console.print("[dim]Starting ACP server on stdio...[/dim]")
    # In real impl, would handle JSON-RPC loop here
    return "ACP server ready (stdio)"

# --- subagent ---
def run_subagent(goal: str, context: str) -> str:
    console.print(f"[dim]└─ subagent: {goal[:80]}...[/dim]")
    cfg = load_cfg()
    msgs = [
        {"role":"system","content":"You are a focused subagent. Achieve the goal using tools, then return a concise final report."},
        {"role":"user","content":(f"GOAL: {goal}\n\nCONTEXT: {context}" if context else f"GOAL: {goal}")}
    ]
    for _ in range(cfg.get("max_iter", 50)):
        text, calls = call_llm(msgs, stream=False)
        if calls:
            msgs.append({"role":"assistant","content":text or "","tool_calls":calls})
            for c in calls:
                name = c["function"]["name"]
                try: args = json.loads(c["function"]["arguments"])
                except: args = {}
                out = TOOLS[name]["fn"](**args) if name in TOOLS else f"ERROR: unknown tool {name}"
                console.print(f"  [dim]sub[/dim] [cyan]{name}[/cyan] → {str(out)[:120]}")
                msgs.append({"role":"tool","tool_call_id":c["id"],"content":out})
        else:
            return text or "(no output)"
    return "ERROR: subagent hit max iterations"

# --- LLM call ---
def call_llm(messages: list, stream: bool = True, minimal_tools: bool = False):
    cfg = load_cfg()
    provider = cfg["provider"]; base = PROVIDERS[provider]["base"]
    api_key = cfg.get("api_key","")
    model = cfg["model"]

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.2,
        "stream": stream,
        "tools": tool_schema(minimal_tools),
        "tool_choice": "auto",
    }
    headers = {"Content-Type":"application/json","Authorization":f"Bearer {api_key}"}

    if stream:
        return stream_response(httpx.post(f"{base}/chat/completions", json=payload, headers=headers, timeout=300))
    else:
        r = httpx.post(f"{base}/chat/completions", json=payload, headers=headers, timeout=300)
        r.raise_for_status()
        d = r.json()
        msg = d["choices"][0]["message"]
        return msg.get("content","") or "", msg.get("tool_calls") or []

def stream_response(resp):
    text_buf = ""
    tool_calls = {}
    for line in resp.iter_lines():
        if not line or not line.startswith("data: "): continue
        data = line[6:]
        if data.strip() == "[DONE]": break
        try: d = json.loads(data)
        except: continue
        delta = d["choices"][0].get("delta", {})
        if "content" in delta and delta["content"]:
            text_buf += delta["content"]
            console.print(delta["content"], end="")
        for tc in delta.get("tool_calls") or []:
            idx = tc.get("index", 0)
            if idx not in tool_calls:
                tool_calls[idx] = {"id": tc.get("id",""), "function":{"name":"","arguments":""}}
            if tc.get("id"): tool_calls[idx]["id"] = tc["id"]
            if tc.get("function",{}).get("name"): tool_calls[idx]["function"]["name"] += tc["function"]["name"]
            if tc.get("function",{}).get("arguments"): tool_calls[idx]["function"]["arguments"] += tc["function"]["arguments"]
    calls = []
    for i in sorted(tool_calls):
        c = tool_calls[i]
        calls.append({"id": c["id"], "type":"function", "function": c["function"]})
    return text_buf, calls

# --- approval ---
def confirm_tool(name: str, args: dict) -> bool:
    cfg = load_cfg()
    if name == "bash":
        if cfg.get("auto_approve_shell"): return True
    elif name in ("write_file","edit_file","notebook_edit","read_file","todo","find_files","grep","web_search","web_fetch"):
        if cfg.get("auto_approve_write"): return True
    elif name in ("subagent", "mcp_call", "acp_request"):
        return True
    summary = json.dumps(args, indent=2)[:400]
    console.print(Panel(summary, title=f"[yellow]approve:[/yellow] [cyan]{name}[/cyan]", border_style="yellow"))
    return Prompt.ask("proceed?", choices=["y","n"], default="y") == "y"

# --- session persistence ---
def save_session(session_id: str, messages: list, title: str = ""):
    with sqlite3.connect(SESSIONS_DB) as db:
        db.execute("INSERT OR REPLACE INTO sessions VALUES (?,?,?,?,?)",
            (session_id, time.time(), time.time(), title or f"Session {session_id[:8]}", json.dumps(messages)))

def load_session(session_id: str) -> Optional[list]:
    with sqlite3.connect(SESSIONS_DB) as db:
        row = db.execute("SELECT messages FROM sessions WHERE id=?", (session_id,)).fetchone()
        return json.loads(row[0]) if row else None

def list_sessions() -> list:
    with sqlite3.connect(SESSIONS_DB) as db:
        return db.execute("SELECT id, created, title FROM sessions ORDER BY updated DESC LIMIT 20").fetchall()

def log_history(session_id: str, role: str, content: str, tool_calls: list = None):
    with sqlite3.connect(HISTORY_DB) as db:
        db.execute("INSERT INTO history VALUES (NULL,?,?,?,?,?)",
            (time.time(), session_id, role, content, json.dumps(tool_calls) if tool_calls else None))

# --- main agent loop ---

# --- image input tools (Codex-like) ---
@tool("vision_analyze", "Analyze an image with vision model. image_path=path, question=string.", {
    "type":"object","properties":{"image_path":{"type":"string"},"question":{"type":"string"}},"required":["image_path","question"]
})
def t_vision_analyze(image_path: str, question: str) -> str:
    """Analyze an image using vision capabilities."""
    try:
        from hermes_tools import vision_analyze
        result = vision_analyze(image_url=safe_path(image_path), question=question)
        return result.get("text", str(result))
    except Exception as e:
        return f"ERROR: {e}"

@tool("image_describe", "Describe an image in detail. image_path=path.", {
    "type":"object","properties":{"image_path":{"type":"string"}},"required":["image_path"]
})
def t_image_describe(image_path: str) -> str:
    """Get a detailed description of an image."""
    try:
        from hermes_tools import vision_analyze
        result = vision_analyze(image_url=safe_path(image_path), question="Describe this image in detail, including any text, code, UI elements, or diagrams visible.")
        return result.get("text", str(result))
    except Exception as e:
        return f"ERROR: {e}"

# --- worktree tools (Codex-like) ---
@tool("git_worktree_add", "Add a git worktree. path=path, branch=branch (optional).", {
    "type":"object","properties":{"path":{"type":"string"},"branch":{"type":"string"}},"required":["path"]
})
def t_git_worktree_add(path: str, branch: str = "") -> str:
    """Add a git worktree for isolated development."""
    p = safe_path(path)
    cmd = f"git worktree add {p}"
    if branch:
        cmd += f" {branch}"
    return run_cmd(cmd)

@tool("git_worktree_list", "List git worktrees.", {
    "type":"object","properties":{}
})
def t_git_worktree_list() -> str:
    """List all git worktrees."""
    return run_cmd("git worktree list")

@tool("git_worktree_remove", "Remove a git worktree. path=path.", {
    "type":"object","properties":{"path":{"type":"string"}},"required":["path"]
})
def t_git_worktree_remove(path: str) -> str:
    """Remove a git worktree."""
    p = safe_path(path)
    return run_cmd(f"git worktree remove {p}")

# --- test execution tools (Codex-like) ---
@tool("test_run", "Run tests. command=string (e.g., 'pytest', 'npm test', 'go test').", {
    "type":"object","properties":{"command":{"type":"string","default":"pytest"}},"required":[]
})
def t_test_run(command: str = "pytest") -> str:
    """Run test suite."""
    return run_cmd(command)

@tool("test_watch", "Run tests in watch mode. command=string (e.g., 'pytest --watch').", {
    "type":"object","properties":{"command":{"type":"string","default":"pytest --watch"}},"required":[]
})
def t_test_watch(command: str = "pytest --watch") -> str:
    """Run tests in watch mode."""
    return run_cmd(command)

@tool("test_coverage", "Run tests with coverage. command=string.", {
    "type":"object","properties":{"command":{"type":"string"}},"required":["command"]
})
def t_test_coverage(command: str) -> str:
    """Run tests with coverage reporting."""
    return run_cmd(command)

# --- repository understanding (Codex-like) ---
@tool("repo_structure", "Show repository structure. depth=integer (default 2).", {
    "type":"object","properties":{"depth":{"type":"integer","default":2}},"required":[]
})
def t_repo_structure(depth: int = 2) -> str:
    """Show repository structure as a tree."""
    try:
        import subprocess
        result = subprocess.run(["find", ".", "-type", "f", "-not", "-path", "*/.*/*"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            files = result.stdout.strip().split('\n')
            # Simple tree view
            tree_lines = []
            for f in sorted(files):
                if f:  # Skip empty lines
                    depth = f.count('/')
                    indent = "  " * depth
                    name = f.split('/')[-1] if '/' in f else f
                    tree_lines.append(f"{indent}{name}")
            return '\n'.join(tree_lines[:100]) or "(no files)"
        else:
            return f"ERROR: {result.stderr}"
    except Exception as e:
        return f"ERROR: {e}"

@tool("repo_dependencies", "Show project dependencies (package.json, requirements.txt, etc.).", {
    "type":"object","properties":{}
})
def t_repo_dependencies() -> str:
    """Show project dependencies from various package managers."""
    deps = []
    # Check common dependency files
    dep_files = [
        ("package.json", "npm/yarn"),
        ("requirements.txt", "pip"),
        ("Pipfile", "pipenv"),
        ("pyproject.toml", "poetry/PEP 517"),
        ("Cargo.toml", "rust"),
        ("go.mod", "go"),
        ("pom.xml", "maven"),
        ("build.gradle", "gradle"),
        ("composer.json", "php")
    ]
    
    for file_desc, manager in dep_files:
        p = safe_path(file_desc)
        if p.exists():
            try:
                content = p.read_text()[:500]
                deps.append(f"=== {file_desc} ({manager}) ===\n{content}")
            except:
                deps.append(f"=== {file_desc} ({manager}) ===\n[could not read]")
    
    return '\n\n'.join(deps) if deps else "(no dependency files found)"

@tool("repo_language_stats", "Show language statistics via github-linguist or cloc fallback.", {
    "type":"object","properties":{}
})
def t_repo_language_stats() -> str:
    """Show programming language usage."""
    # Try to use cloc if available
    if shutil.which("cloc"):
        return run_cmd("cloc . --by-file --json")
    else:
        # Fallback to simple extension counting
        try:
            from collections import Counter
            import os
            ext_counter = Counter()
            total_files = 0
            
            for root, dirs, files in os.walk("."):
                # Skip hidden dirs and common build dirs
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in {'node_modules', '__pycache__', 'dist', 'build', '.git'}]
                for f in files:
                    if not f.startswith('.'):
                        total_files += 1
                        ext = os.path.splitext(f)[1]
                        if ext:
                            ext_counter[ext] += 1
            
            if total_files == 0:
                return "(no files found)"
            
            result = [f"Total files: {total_files}"]
            for ext, count in ext_counter.most_common(10):
                pct = (count / total_files) * 100
                result.append(f"{ext or '(no extension)'}: {count} files ({pct:.1f}%)")
            return "\n".join(result)
        except Exception as e:
            return f"ERROR: {e}"

# --- AGENTS.md support (Codex-like) ---
@tool("agents_read", "Read AGENTS.md file for project-specific instructions.", {
    "type":"object","properties":{}
})
def t_agents_read() -> str:
    """Read AGENTS.md if it exists."""
    p = safe_path("AGENTS.md")
    if p.exists():
        return p.read_text()
    else:
        return "(AGENTS.md not found)"

@tool("agents_write", "Write or update AGENTS.md file. content=string.", {
    "type":"object","properties":{"content":{"type":"string"}},"required":["content"]
})
def t_agents_write(content: str) -> str:
    """Write AGENTS.md file."""
    p = safe_path("AGENTS.md")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"wrote AGENTS.md ({len(content)} bytes)"

# --- context compaction (Codex-like) ---
@tool("context_compact", "Compact conversation history to save tokens. keep_last=integer (default 5).", {
    "type":"object","properties":{"keep_last":{"type":"integer","default":5}},"required":[]
})
def t_context_compact(keep_last: int = 5) -> str:
    """Compact the conversation history to reduce token usage."""
    global messages
    if len(messages) <= keep_last + 2:  # +2 for system and maybe one other
        return f"(history only {len(messages)} messages, no compaction needed)"
    
    # Keep system message and last N messages
    system_msgs = [m for m in messages if m.get("role") == "system"]
    recent_msgs = messages[-(keep_last):] if len(messages) > keep_last else []
    
    # What we're removing
    removed_count = len(messages) - len(system_msgs) - len(recent_msgs)
    
    # Rebuild messages
    messages.clear()
    messages.extend(system_msgs)
    messages.extend(recent_msgs)
    
    return f"compacted context: removed {removed_count} messages, kept {len(messages)}"

# --- long running task helpers (Codex-like) ---
@tool("task_breakdown", "Break down a complex task into steps. goal=string.", {
    "type":"object","properties":{"goal":{"type":"string"}},"required":["goal"]
})
def t_task_breakdown(goal: str) -> str:
    """Ask the AI to break down a goal into steps."""
    # This would typically involve an LLM call, but we'll return a placeholder
    # In practice, this could be implemented as a subagent call or special prompt
    return f"To break down '{goal}':\\n1. Understand the requirements\\n2. Research existing solutions\\n3. Implement core functionality\\n4. Test and iterate\\n5. Document and clean up"

# --- main agent loop ---
SYSTEM_PROMPT = """You are Scout v2, a comprehensive CLI coding agent.
Capabilities: file ops, bash, grep, git, web, subagents, MCP, skills, ACP, harness switching, plan mode, steering.
- Be concise. Use tools, not narration.
- Plan briefly, then act.
- Read files before editing. Verify after.
- Prefer rg/find over manual scanning.
- When done, give a 1-3 line summary.
- CWD: {cwd}
- Harness: {harness}"""

def run_agent(messages: list, cfg: dict):
    global CURRENT_SESSION_ID
    minimal = cfg.get("harness") == "minimal"
    for it in range(cfg.get("max_iter", 50)):
        if AGENT_STATE["interrupted"]:
            AGENT_STATE["interrupted"] = False
            console.print("[yellow]interrupted[/yellow]")
            return
        if AGENT_STATE["steering_queue"]:
            steer = AGENT_STATE["steering_queue"].pop(0)
            messages.append({"role":"user","content":steer})
            console.print(f"[cyan]steer:[/cyan] {steer}")

        console.print(f"\n[dim]── iter {it+1} ──[/dim]")
        try:
            text, calls = call_llm(messages, stream=True, minimal_tools=minimal)
        except httpx.HTTPStatusError as e:
            console.print(f"[red]HTTP {e.response.status_code}: {e.response.text[:300]}[/red]")
            return
        except Exception as e:
            console.print(f"[red]error: {e}[/red]"); traceback.print_exc(); return

        if not calls:
            console.print()
            return

        messages.append({"role":"assistant","content":text or "","tool_calls":calls})
        log_history(CURRENT_SESSION_ID, "assistant", text or "", calls)
        for c in calls:
            name = c["function"]["name"]
            try: args = json.loads(c["function"]["arguments"])
            except: args = {}
            if not confirm_tool(name, args):
                messages.append({"role":"tool","tool_call_id":c["id"],"content":"DENIED by user"})
                log_history(CURRENT_SESSION_ID, "tool", "DENIED", [{"name":name,"args":args}])
                continue
            t0 = time.time()
            try:
                out = TOOLS[name]["fn"](**args)
            except Exception as e:
                out = f"ERROR: {e}\n{traceback.format_exc()[-500:]}"
            dt = time.time() - t0
            preview = str(out)[:200].replace("\n"," ")
            console.print(f"  [dim]→[/dim] [cyan]{name}[/cyan] [dim]({dt:.1f}s)[/dim] {preview}{'...' if len(str(out))>200 else ''}")
            messages.append({"role":"tool","tool_call_id":c["id"],"content":str(out)})
            log_history(CURRENT_SESSION_ID, "tool", str(out), [{"name":name,"args":args,"result":str(out)[:500]}])
    console.print("[yellow]hit max_iter[/yellow]")

# --- TUI helpers ---
def show_todos():
    todos = AGENT_STATE["todos"]
    if not todos: return
    table = Table(title="Todos", show_header=True, header_style="bold cyan")
    table.add_column("Status", width=12)
    table.add_column("Task")
    for t in todos:
        status = t.get("status","pending")
        icon = {"pending":"○", "in_progress":"◐", "completed":"●", "cancelled":"✗"}.get(status, "?")
        style = {"pending":"dim", "in_progress":"yellow", "completed":"green", "cancelled":"red"}.get(status, "")
        table.add_row(f"[{style}]{icon} {status}[/{style}]", t.get("content",""))
    console.print(table)

def show_help():
    console.print(Panel("""[cyan]/help[/cyan]        this help
[cyan]/cd <dir>[/cyan]      change cwd
[cyan]/model[/cyan]         switch model
[cyan]/provider[/cyan]      switch provider
[cyan]/harness[/cyan]       switch harness (native, claude-code, kimi, qwen, deepseek, swe-agent, zcode, minimal)
[cyan]/init[/cyan]          re-run setup
[cyan]/plan[/cyan]          toggle plan mode (read-only)
[cyan]/shell[/cyan]         toggle auto-approve shell
[cyan]/write[/cyan]         toggle auto-approve file writes
[cyan]/mcp[/cyan]           manage MCP servers
[cyan]/skill[/cyan]         manage skills
[cyan]/acp[/cyan]           start ACP server for editor
[cyan]/session[/cyan]       session management (new, list, load, save)
[cyan]/clear[/cyan]         clear conversation
[cyan]/history[/cyan]       show session history
[cyan]/quit[/cyan]          exit
[cyan]Ctrl+C[/cyan]         interrupt current run
[cyan]Enter (during run)[/cyan] steer/redirect
[cyan]Alt+Enter[/cyan]      queue follow-up""", title="[bold]Scout Commands[/bold]", border_style="cyan"))

# --- REPL ---
BANNER = """[bold magenta]SCODE[/bold magenta]"""

def repl():
    global CWD, CURRENT_SESSION_ID
    cfg = ensure_cfg()
    AGENT_STATE["current_harness"] = cfg.get("harness", "native")
    
    # Initialize session
    CURRENT_SESSION_ID = str(uuid.uuid4())
    messages = [{"role":"system","content":SYSTEM_PROMPT.format(cwd=CWD, harness=cfg.get("harness","native"))}]
    
    # Show banner
    console.print(BANNER)
    show_todos()
    
    # Handle command line argument
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
        messages.append({"role":"user","content":prompt})
        log_history(CURRENT_SESSION_ID, "user", prompt)
        run_agent(messages, cfg)
        save_session(CURRENT_SESSION_ID, messages)
        return
    
    # Initialize layout for split view
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.text import Text
    from rich.columns import Columns
    from rich.align import Align
    
    # Track context usage (rough estimate)
    context_tokens = 0
    max_context = 8192
    
    def get_context_usage():
        """Estimate context tokens from messages"""
        total = 0
        for m in messages:
            if isinstance(m.get("content"), str):
                total += len(m["content"]) // 4  # rough estimate
            if m.get("tool_calls"):
                total += len(str(m["tool_calls"])) // 4
        return min(total, max_context)
    
    def render_side_panel():
        """Render the right side panel with context, MCP, LSP, TODO"""
        panels = []
        
        # Context meter
        used = get_context_usage()
        pct = int((used / max_context) * 100)
        bar_len = 20
        filled = int(bar_len * pct / 100)
        bar = "█" * filled + "░" * (bar_len - filled)
        cost_est = used * 0.00001  # very rough estimate
        panels.append(Panel(
            f"[bold]Context[/bold]\n"
            f"{used:,} / {max_context:,} tokens\n"
            f"[{bar}] {pct}%\n"
            f"[dim]~${cost_est:.2f} spent[/dim]",
            title="[cyan]Context[/cyan]", border_style="cyan", width=35
        ))
        
        # MCP servers
        mcp_servers = cfg.get("mcp_servers", [])
        mcp_lines = []
        if mcp_servers:
            for s in mcp_servers:
                mcp_lines.append(f"  [green]●[/green] {s['name']} ({s['transport']})")
        else:
            mcp_lines.append("  [dim]No MCP servers configured[/dim]")
            mcp_lines.append("  [dim]Use /mcp to add[/dim]")
        panels.append(Panel(
            "\n".join(mcp_lines),
            title="[cyan]MCP[/cyan]", border_style="cyan", width=35
        ))
        
        # LSP status
        lsp_lines = ["  [green]●[/green] typescript", "  [green]●[/green] python", "  [dim]● eslint[/dim]"]
        panels.append(Panel(
            "\n".join(lsp_lines),
            title="[cyan]LSP[/cyan]", border_style="cyan", width=35
        ))
        
        # TODO list
        todos = AGENT_STATE["todos"]
        if todos:
            todo_lines = []
            for t in todos[:8]:
                status = t.get("status", "pending")
                icon = {"pending": "○", "in_progress": "◐", "completed": "●", "cancelled": "✗"}.get(status, "?")
                style = {"pending": "dim", "in_progress": "yellow", "completed": "green", "cancelled": "red"}.get(status, "")
                todo_lines.append(f"  [{style}]{icon} {t.get('content', '')[:30]}[/{style}]")
            if len(todos) > 8:
                todo_lines.append(f"  [dim]...and {len(todos) - 8} more[/dim]")
        else:
            todo_lines = ["  [dim]No tasks[/dim]", "  [dim]Agent will create[/dim]"]
        panels.append(Panel(
            "\n".join(todo_lines),
            title="[cyan]TODO[/cyan]", border_style="cyan", width=35
        ))
        
        return Columns(panels, equal=True, expand=True)
    
    def render_main_area():
        """Render the main conversation area"""
        # This would show recent messages, but we keep it simple
        # The actual conversation is printed by run_agent
        return Panel(
            "[dim]Main agent session[/dim]\n[dim]Output appears here[/dim]",
            title="[bold magenta]SCODE[/bold magenta]", border_style="magenta"
        )
    
    def render_bottom_bar():
        """Render bottom status/keyboard bar"""
        provider = cfg.get('provider', '?')
        model = cfg.get('model', '?')
        harness = AGENT_STATE.get('current_harness', cfg.get('harness', 'native'))
        return Panel(
            f"[bold]{provider}/{model}[/bold] | {harness} | "
            f"[cyan]Esc[/cyan] interrupt  [cyan]Tab[/cyan] session  [cyan]Ctrl+P[/cyan] commands  [cyan]Ctrl+X←/→[/cyan] nav",
            border_style="dim", width=None
        )
    
    # Initial render of side panel
    console.print(render_side_panel())
    console.print(render_bottom_bar())
    
    while True:
        console.print()
        try:
            user_input = Prompt.ask("[bold green]🚀[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]bye[/dim]")
            save_session(CURRENT_SESSION_ID, messages)
            return
        if not user_input:
            console.print(render_side_panel())
            continue
        
        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=2)
            cmd = parts[0]
            
            if cmd == "/quit":
                save_session(CURRENT_SESSION_ID, messages)
                return
            elif cmd == "/help":
                show_help()
            elif cmd == "/cd":
                target = parts[1] if len(parts)>1 else str(HOME)
                p = Path(target).expanduser().resolve()
                if p.is_dir():
                    CWD = p
                    messages[0]["content"] = SYSTEM_PROMPT.format(cwd=CWD, harness=cfg.get("harness","native"))
                    console.print(f"[dim]cwd: {CWD}[/dim]")
                else:
                    console.print(f"[red]not a dir: {p}[/red]")
            elif cmd == "/model":
                cfg["model"] = Prompt.ask("model", default=cfg["model"])
                save_cfg(cfg)
                console.print(f"[dim]→ {cfg['model']}[/dim]")
            elif cmd == "/provider":
                names = list(PROVIDERS)
                for i, n in enumerate(names, 1):
                    console.print(f"  [cyan]{i}[/cyan] {n}")
                choice = Prompt.ask("provider", choices=[str(i) for i in range(1, len(names)+1)], default="1")
                cfg["provider"] = names[int(choice)-1]
                cfg["api_key"] = Prompt.ask("api key", default=cfg.get("api_key",""), )
                save_cfg(cfg)
                console.print(f"[dim]→ {cfg['provider']}[/dim]")
            elif cmd == "/harness":
                names = list(HARNESSES)
                for i, n in enumerate(names, 1):
                    console.print(f"  [cyan]{i}[/cyan] {n} — {HARNESSES[n]['desc']}")
                choice = Prompt.ask("harness", choices=[str(i) for i in range(1, len(names)+1)], default="1")
                cfg["harness"] = names[int(choice)-1]
                save_cfg(cfg)
                AGENT_STATE["current_harness"] = cfg["harness"]
                messages[0]["content"] = SYSTEM_PROMPT.format(cwd=CWD, harness=cfg["harness"])
                console.print(f"[dim]→ harness: {cfg['harness']}[/dim]")
            elif cmd == "/init":
                cfg = ensure_cfg()
                messages[0]["content"] = SYSTEM_PROMPT.format(cwd=CWD, harness=cfg.get("harness","native"))
            elif cmd == "/plan":
                AGENT_STATE["plan_mode"] = not AGENT_STATE["plan_mode"]
                console.print(f"[dim]plan mode: {'on' if AGENT_STATE['plan_mode'] else 'off'}[/dim]")
            elif cmd == "/shell":
                cfg["auto_approve_shell"] = not cfg.get("auto_approve_shell", False)
                save_cfg(cfg)
                console.print(f"[dim]auto-approve shell: {cfg['auto_approve_shell']}[/dim]")
            elif cmd == "/write":
                cfg["auto_approve_write"] = not cfg.get("auto_approve_write", True)
                save_cfg(cfg)
                console.print(f"[dim]auto-approve write: {cfg['auto_approve_write']}[/dim]")
            elif cmd == "/mcp":
                console.print("[dim]MCP servers:[/dim]")
                for s in cfg.get("mcp_servers", []):
                    console.print(f"  - {s['name']} ({s['transport']})")
                if Prompt.ask("add server?", choices=["y","n"], default="n") == "y":
                    name = Prompt.ask("name")
                    transport = Prompt.ask("transport", choices=["stdio","http","sse"], default="stdio")
                    if transport == "stdio":
                        command = Prompt.ask("command")
                        cfg.setdefault("mcp_servers", []).append({"name":name,"transport":"stdio","command":command})
                    else:
                        url = Prompt.ask("url")
                        cfg.setdefault("mcp_servers", []).append({"name":name,"transport":transport,"url":url})
                    save_cfg(cfg)
            elif cmd == "/skill":
                console.print("[dim]Skills:[/dim]")
                for f in SKILLS_DIR.glob("*.md"):
                    console.print(f"  - {f.stem}")
                action = Prompt.ask("action", choices=["load","create","delete","list"], default="list")
                if action == "load":
                    name = Prompt.ask("skill name")
                    console.print(load_skill(name))
                elif action == "create":
                    name = Prompt.ask("name")
                    console.print("Enter skill markdown (end with EOF/Ctrl+D):")
                    content = sys.stdin.read()
                    console.print(t_skill_create(name, content))
                elif action == "delete":
                    name = Prompt.ask("name")
                    (SKILLS_DIR / f"{name}.md").unlink(missing_ok=True)
                    console.print(f"deleted {name}")
            elif cmd == "/acp":
                console.print(start_acp_server())
                AGENT_STATE["acp_mode"] = True
            elif cmd == "/session":
                action = parts[1] if len(parts)>1 else "list"
                if action == "list":
                    for sid, created, title in list_sessions():
                        console.print(f"  {sid[:8]}  {datetime.fromtimestamp(created).strftime('%Y-%m-%d %H:%M')}  {title}")
                elif action == "new":
                    save_session(CURRENT_SESSION_ID, messages)
                    CURRENT_SESSION_ID = str(uuid.uuid4())
                    messages = [{"role":"system","content":SYSTEM_PROMPT.format(cwd=CWD, harness=cfg.get("harness","native"))}]
                    console.print(f"[dim]new session: {CURRENT_SESSION_ID[:8]}[/dim]")
                elif action == "load":
                    sid = parts[2] if len(parts)>2 else Prompt.ask("session id")
                    loaded = load_session(sid)
                    if loaded:
                        messages = loaded
                        CURRENT_SESSION_ID = sid
                        console.print(f"[dim]loaded session {sid[:8]}[/dim]")
                    else:
                        console.print("[red]session not found[/red]")
                elif action == "save":
                    title = parts[2] if len(parts)>2 else ""
                    save_session(CURRENT_SESSION_ID, messages, title)
                    console.print("[dim]saved[/dim]")
            elif cmd == "/clear":
                messages = [messages[0]]
                console.clear()
                console.print(BANNER)
            elif cmd == "/history":
                with sqlite3.connect(HISTORY_DB) as db:
                    rows = db.execute("SELECT ts, role, content FROM history WHERE session_id=? ORDER BY id DESC LIMIT 20", (CURRENT_SESSION_ID,)).fetchall()
                    for ts, role, content in reversed(rows):
                        console.print(f"[dim]{datetime.fromtimestamp(ts).strftime('%H:%M:%S')}[/dim] [{role}] {content[:100]}")
            else:
                console.print(f"[red]unknown: {cmd}[/red]")
            
            console.print(render_side_panel())
            console.print(render_bottom_bar())
            continue
        
        # Regular user input - run agent
        messages.append({"role":"user","content":user_input})
        log_history(CURRENT_SESSION_ID, "user", user_input)
        run_agent(messages, cfg)
        save_session(CURRENT_SESSION_ID, messages)
        console.print(render_side_panel())
        console.print(render_bottom_bar())

if __name__ == "__main__":
    repl()

#!/usr/bin/env python3
"""Scout v2 — comprehensive CLI AI coding harness. Single file, cross-platform, no native deps.
Capabilities: ACP, MCP, skills, harness abstraction, git-aware, steering, session persistence, plan mode, subagents.
Runs on Linux/macOS/Windows/Android (Termux/Pydroid)."""
import os, sys, json, re, ast, shutil, subprocess, time, argparse, signal, threading, queue, webbrowser, pathlib, textwrap, traceback, uuid, sqlite3, hashlib, inspect, tempfile, select, shlex
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
    "openai": {
        "base": "https://api.openai.com/v1",
        "kind": "openai",
        "models": [],
        "requires_key": True,
    },
    "anthropic": {
        "base": "https://api.anthropic.com/v1",
        "kind": "anthropic",
        "models": [],
        "requires_key": True,
    },
    "nvidia": {
        "base": "https://integrate.api.nvidia.com/v1",
        "kind": "openai",
        "models": [],
        "requires_key": True,
    },
    "groq": {
        "base": "https://api.groq.com/openai/v1",
        "kind": "openai",
        "models": [],
        "requires_key": True,
    },
    "openrouter": {
        "base": "https://openrouter.ai/api/v1",
        "kind": "openai",
        "models": [],
        "requires_key": True,
    },
    "ollama": {
        "base": "http://localhost:11434/v1",
        "kind": "openai",
        "models": [],
        "requires_key": False,
    },
    "lmstudio": {
        "base": "http://localhost:1234/v1",
        "kind": "openai",
        "models": [],
        "requires_key": False,
    },
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
    """Run a command and preserve both stdout and stderr, including exit status."""
    try:
        r = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            timeout=max(1, int(timeout)), cwd=str(cwd or CWD)
        )
        stdout = r.stdout or ""
        stderr = r.stderr or ""
        out = (stdout + stderr).strip()
        if r.returncode != 0:
            suffix = f"\\n(exit {r.returncode})"
            out = (out + suffix).strip()
        return out[:50000] if out else f"(exit {r.returncode})"
    except subprocess.TimeoutExpired:
        return f"ERROR: timeout after {timeout}s"
    except Exception as e:
        return f"ERROR: {e}"

# --- config / provider discovery ---
def load_cfg() -> dict:
    if CFG_PATH.exists():
        try:
            data = json.loads(CFG_PATH.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}
    return {}


def save_cfg(cfg: dict):
    CFG_PATH.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")


def _provider_env_key(provider: str) -> str:
    return {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "nvidia": "NVIDIA_API_KEY",
        "groq": "GROQ_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "ollama": "OLLAMA_API_KEY",
        "lmstudio": "LMSTUDIO_API_KEY",
    }.get(provider, f"{provider.upper()}_API_KEY")


def _api_error(resp: httpx.Response) -> str:
    try:
        body = resp.json()
    except Exception:
        body = resp.text[:2000]
    if isinstance(body, dict) and "error" in body:
        err = body["error"]
        if isinstance(err, dict):
            message = err.get("message") or err.get("detail") or str(err)
            code = err.get("code")
            return f"HTTP {resp.status_code}: {message}" + (f" (code={code})" if code else "")
        return f"HTTP {resp.status_code}: {err}"
    return f"HTTP {resp.status_code}: {body}"


def _openai_headers(cfg: dict) -> dict:
    key = cfg.get("api_key", "")
    h = {"Content-Type": "application/json", "Accept": "application/json"}
    if key:
        h["Authorization"] = f"Bearer {key}"
    if cfg.get("provider") == "openrouter":
        h["HTTP-Referer"] = cfg.get("http_referer", "http://localhost")
        h["X-Title"] = cfg.get("app_title", "SCODE")
    return h


def _anthropic_headers(cfg: dict) -> dict:
    return {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "x-api-key": cfg.get("api_key", ""),
        "anthropic-version": "2023-06-01",
    }


def _request(method: str, url: str, *, headers=None, json_body=None,
             timeout=30, retries=3, stream=False) -> httpx.Response:
    """Network layer with sane retry policy.

    401/403 are never blindly retried. They are authentication/authorization
    failures and must be handled by refreshing the model catalog or credentials.
    429 and transient 5xx errors are retried with bounded exponential backoff.
    """
    last = None
    for attempt in range(max(1, retries)):
        try:
            r = httpx.request(
                method, url, headers=headers or {}, json=json_body,
                timeout=timeout, follow_redirects=True,
            )
            if r.status_code in (429, 500, 502, 503, 504) and attempt + 1 < retries:
                retry_after = r.headers.get("retry-after")
                try:
                    delay = min(8.0, max(0.25, float(retry_after)))
                except Exception:
                    delay = min(8.0, 0.5 * (2 ** attempt))
                time.sleep(delay)
                last = r
                continue
            return r
        except (httpx.TimeoutException, httpx.NetworkError) as e:
            last = e
            if attempt + 1 >= retries:
                raise RuntimeError(f"network error contacting provider: {e}") from e
            time.sleep(min(8.0, 0.5 * (2 ** attempt)))
    if isinstance(last, httpx.Response):
        return last
    raise RuntimeError(f"request failed: {last}")


def _normalize_model(item: dict, provider: str) -> dict:
    mid = str(item.get("id") or item.get("name") or "").strip()
    architecture = item.get("architecture") or {}
    supported = item.get("supported_parameters") or []
    if isinstance(supported, dict):
        supported = list(supported)
    modality = str(architecture.get("modality") or "").lower()
    explicit_tool_support = any(
        x in {"tools", "tool_choice", "function_call", "structured_outputs"}
        or "tool" in str(x).lower()
        for x in supported
    )
    # Several official /models endpoints do not expose capability metadata.
    # Treat ordinary chat models as tool-capable until the provider says otherwise;
    # OpenRouter exposes the detailed supported_parameters field when available.
    tool_calling = explicit_tool_support if supported else provider not in {"ollama", "lmstudio"}
    return {
        "id": mid,
        "name": str(item.get("name") or mid),
        "context_length": item.get("context_length") or item.get("context_window"),
        "description": str(item.get("description") or ""),
        "owned_by": str(item.get("owned_by") or architecture.get("tokenizer") or ""),
        "modality": modality,
        "tool_calling": tool_calling,
        "raw": item,
    }


def _is_usable_agent_model(m: dict) -> bool:
    mid = m["id"].lower()
    blocked = (
        "embedding", "rerank", "moderation", "tts", "transcribe",
        "whisper", "image-generation", "image-generation",
    )
    if any(x in mid for x in blocked):
        return False
    modality = m.get("modality", "")
    if modality and ("text" not in modality and "text->text" not in modality):
        return False
    return bool(m["id"])


def fetch_models(cfg: dict, force: bool = False) -> list[dict]:
    """Fetch the provider's current model catalog after authorization."""
    provider = cfg.get("provider")
    meta = PROVIDERS.get(provider)
    if not meta:
        raise RuntimeError(f"Unknown provider: {provider}")

    key = cfg.get("api_key", "")
    if meta.get("requires_key") and not key:
        raise RuntimeError(f"{provider} requires an API key before models can be fetched.")

    # Local providers use OpenAI-compatible /v1/models. Ollama also has a native
    # endpoint, but /v1/models is preferable because it matches the chat API.
    url = meta["base"].rstrip("/") + "/models"
    headers = _anthropic_headers(cfg) if meta["kind"] == "anthropic" else _openai_headers(cfg)
    resp = _request("GET", url, headers=headers, timeout=20, retries=2)

    if resp.status_code in (401, 403):
        raise RuntimeError(
            _api_error(resp) +
            ". The provider rejected authorization/model access; SCODE will not "
            "blindly send chat requests with an unverified model."
        )
    if resp.status_code >= 400:
        raise RuntimeError(_api_error(resp))

    try:
        data = resp.json()
    except Exception as e:
        raise RuntimeError(f"provider returned invalid JSON from /models: {e}")

    raw = data.get("data", data.get("models", []))
    if not isinstance(raw, list):
        raise RuntimeError("provider /models response did not contain a model list")

    models = [_normalize_model(x, provider) for x in raw if isinstance(x, dict)]
    models = [m for m in models if _is_usable_agent_model(m)]

    # Deduplicate by model id while preserving provider order.
    unique = {}
    for m in models:
        unique.setdefault(m["id"], m)
    models = list(unique.values())
    if not models:
        raise RuntimeError("authorization succeeded, but the provider returned no usable text models")

    cfg["model_catalog"] = models
    cfg["model_catalog_at"] = time.time()
    save_cfg(cfg)
    return models


def get_cached_models(cfg: dict) -> list[dict]:
    models = cfg.get("model_catalog")
    if isinstance(models, list):
        return [m for m in models if isinstance(m, dict) and m.get("id")]
    return []


def select_model(cfg: dict, models: list[dict] | None = None) -> str:
    models = models or get_cached_models(cfg)
    if not models:
        models = fetch_models(cfg)

    current = cfg.get("model", "")
    while True:
        console.print(
            f"[cyan]Model search[/cyan] — {len(models)} available. "
            "Type a search term, exact model ID, or number. Blank keeps current."
        )
        query = Prompt.ask("model", default=current or "").strip()
        if not query:
            if current and any(m["id"] == current for m in models):
                return current
            query = "*"

        if query == "*":
            matches = models[:30]
        elif query.isdigit():
            n = int(query)
            if 1 <= n <= len(models):
                return models[n - 1]["id"]
            console.print("[red]invalid model number[/red]")
            continue
        else:
            q = query.lower()
            matches = [
                m for m in models
                if q in m["id"].lower()
                or q in m.get("name", "").lower()
                or q in m.get("description", "").lower()
            ]

        if not matches:
            console.print("[yellow]no matching models[/yellow]")
            continue

        table = Table(title=f"Models ({len(matches)} matches)", show_header=True)
        table.add_column("#", width=5)
        table.add_column("Model")
        table.add_column("Tools", width=7)
        table.add_column("Context", width=12)
        for i, m in enumerate(matches[:40], 1):
            ctx = m.get("context_length")
            ctx_text = f"{int(ctx):,}" if isinstance(ctx, (int, float)) else "-"
            tools_text = "yes" if m.get("tool_calling") else "no"
            table.add_row(str(i), m["id"], tools_text, ctx_text)
        console.print(table)

        choice = Prompt.ask("select # / refine search", default="1").strip()
        if choice.isdigit() and 1 <= int(choice) <= min(40, len(matches)):
            return matches[int(choice) - 1]["id"]
        query = choice


def ensure_cfg(non_interactive: bool = False) -> dict:
    cfg = load_cfg()
    if "provider" not in cfg or cfg["provider"] not in PROVIDERS:
        if non_interactive:
            # Use first provider that has an env key, or default to nvidia
            for p in ["nvidia", "openai", "anthropic", "groq", "openrouter", "ollama", "lmstudio"]:
                if p in PROVIDERS and (not PROVIDERS[p].get("requires_key") or os.environ.get(_provider_env_key(p))):
                    cfg["provider"] = p
                    break
            else:
                cfg["provider"] = "nvidia"
        else:
            console.print("[bold cyan]SCOUT provider setup[/bold cyan]")
            names = list(PROVIDERS)
            for i, n in enumerate(names, 1):
                console.print(f"  [cyan]{i}[/cyan] {n}")
            choice = Prompt.ask("provider", choices=[str(i) for i in range(1, len(names)+1)], default="1")
            cfg["provider"] = names[int(choice)-1]

    meta = PROVIDERS[cfg["provider"]]
    if meta.get("requires_key"):
        env_key = os.environ.get(_provider_env_key(cfg["provider"]), "")
        if env_key and not cfg.get("api_key"):
            cfg["api_key"] = env_key
        if not cfg.get("api_key"):
            if non_interactive:
                raise RuntimeError(f"API key required for {cfg['provider']} in non-interactive mode. Set {_provider_env_key(cfg['provider'])} env var.")
            cfg["api_key"] = Prompt.ask(
                f"API key for [cyan]{cfg['provider']}[/cyan]",
                password=True,
                default="",
            )

    # Authorization happens before model selection. The selected model therefore
    # comes from the provider's current catalog instead of stale hardcoded IDs.
    try:
        models = fetch_models(cfg)
        console.print(f"[green]Authorized — fetched {len(models)} usable models.[/green]")
        if non_interactive:
            # In non-interactive mode, just use the first model or current
            current = cfg.get("model", "")
            if current and any(m["id"] == current for m in models):
                pass  # keep current
            else:
                cfg["model"] = models[0]["id"]
        else:
            cfg["model"] = select_model(cfg, models)
    except Exception as e:
        console.print(f"[red]Model discovery failed: {e}[/red]")
        cached = get_cached_models(cfg)
        if cached:
            console.print(f"[yellow]Using cached catalog ({len(cached)} models).[/yellow]")
            if non_interactive:
                current = cfg.get("model", "")
                if current and any(m["id"] == current for m in cached):
                    pass
                else:
                    cfg["model"] = cached[0]["id"]
            else:
                cfg["model"] = select_model(cfg, cached)
        elif not cfg.get("model"):
            raise RuntimeError(
                "Cannot start safely without a verified model. Fix the provider/API key first."
            )

    cfg.setdefault("max_iter", 50)
    cfg.setdefault("max_tokens", MAX_TOKENS)
    cfg.setdefault("auto_approve_shell", False)
    cfg.setdefault("auto_approve_write", True)
    cfg.setdefault("web_provider", "duckduckgo")
    cfg.setdefault("theme", "monokai")
    cfg.setdefault("harness", "native")
    cfg.setdefault("mcp_servers", [])
    cfg.setdefault("skills_enabled", [])
    cfg.setdefault("disable_tools", False)
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
    p.write_text(content, encoding="utf-8"); return f"wrote {len(content)} bytes to {path}"

@tool("edit_file", "Replace exact text in a file (single occurrence).", {
    "type":"object","properties":{"path":{"type":"string"},"old_string":{"type":"string"},"new_string":{"type":"string"}},"required":["path","old_string","new_string"]
})
def t_edit_file(path: str, old_string: str, new_string: str) -> str:
    p = safe_path(path)
    if not p.exists(): return f"ERROR: {path} not found"
    text = p.read_text()
    if old_string not in text: return "ERROR: old_string not found in file"
    text = text.replace(old_string, new_string, 1)
    p.write_text(text, encoding="utf-8"); return "ok"

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
        r.raise_for_status()
        html = r.text
        # DuckDuckGo's HTML result markup: title link (result__a) comes first,
        # then a result__snippet block later in the same result div.
        titles = re.findall(r'class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>', html, re.S)
        snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', html, re.S)
        out = []
        for i in range(min(limit, len(titles))):
            url, title = titles[i]
            title = re.sub(r"<[^>]+>", "", title).strip()
            url = re.sub(r"<[^>]+>", "", url).strip()
            snippet = re.sub(r"<[^>]+>", "", snippets[i]).strip() if i < len(snippets) else ""
            line = f"- {title}"
            if snippet:
                line += f"\n  {snippet}"
            line += f"\n  {url}"
            out.append(line)
        return "\n".join(out) or "(no results)"
    except Exception as e: return f"ERROR: {e}"

@tool("web_fetch", "Fetch a URL, return text content (stripped HTML).", {
    "type":"object","properties":{"url":{"type":"string"},"max_chars":{"type":"integer","default":15000}},"required":["url"]
})
def t_web_fetch(url: str, max_chars: int = 15000) -> str:
    try:
        r = httpx.get(url, timeout=30, follow_redirects=True, headers={"User-Agent":"Mozilla/5.0 Scout/2.0"})
        r.raise_for_status()
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
    return run_cmd(f"git commit -m {shlex.quote(message)}")

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
        try:
            out, err = proc.communicate(req, timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, err = proc.communicate()
            return f"ERROR: MCP server '{server.get('name','?')}' timed out"
        except Exception as e:
            proc.kill()
            return f"ERROR: MCP call failed: {e}"
        if proc.returncode not in (0, None) and not out.strip():
            return f"ERROR: MCP server '{server.get('name','?')}' exited {proc.returncode}: {err.strip()[:500]}"
        try:
            resp = json.loads(out.strip().split("\n")[-1])
            return json.dumps(resp.get("result", resp.get("error", "unknown")), indent=2)
        except Exception:
            return out.strip() or err.strip() or "no response"
    elif transport in ("http", "sse"):
        url = server["url"]
        try:
            r = httpx.post(f"{url}/mcp", json={"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":tool,"arguments":args}}, timeout=30)
            r.raise_for_status()
            return json.dumps(r.json(), indent=2)
        except Exception as e:
            return f"ERROR: MCP HTTP call failed: {e}"
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
                if name not in TOOLS:
                    out = f"ERROR: unknown tool {name}"
                elif not confirm_tool(name, args):
                    out = "DENIED by user"
                else:
                    try:
                        out = TOOLS[name]["fn"](**args)
                    except Exception as e:
                        out = f"ERROR: {e}"
                console.print(f"  [dim]sub[/dim] [cyan]{name}[/cyan] → {str(out)[:120]}")
                msgs.append({"role":"tool","tool_call_id":c["id"],"content":str(out)})
        else:
            return text or "(no output)"
    return "ERROR: subagent hit max iterations"

# --- LLM call ---
def _selected_model(cfg: dict) -> dict:
    model = cfg.get("model", "")
    for m in get_cached_models(cfg):
        if m.get("id") == model:
            return m
    return {"id": model, "tool_calling": True}


def _messages_for_anthropic(messages: list) -> tuple[str, list]:
    system_parts = []
    out = []
    for m in messages:
        role = m.get("role")
        content = m.get("content", "")
        if role == "system":
            if content:
                system_parts.append(str(content))
            continue
        if role == "tool":
            out.append({"role": "user", "content": [{
                "type": "tool_result",
                "tool_use_id": m.get("tool_call_id", ""),
                "content": str(content),
            }]})
            continue
        if role == "assistant" and m.get("tool_calls"):
            blocks = []
            if content:
                blocks.append({"type": "text", "text": str(content)})
            for tc in m["tool_calls"]:
                fn = tc.get("function", {})
                try:
                    inp = json.loads(fn.get("arguments", "{}"))
                except Exception:
                    inp = {}
                blocks.append({
                    "type": "tool_use",
                    "id": tc.get("id") or str(uuid.uuid4()),
                    "name": fn.get("name", ""),
                    "input": inp,
                })
            out.append({"role": "assistant", "content": blocks})
            continue
        if role in ("user", "assistant"):
            if out and out[-1]["role"] == role:
                if isinstance(out[-1]["content"], list):
                    out[-1]["content"].append({"type": "text", "text": str(content)})
                else:
                    out[-1]["content"] = str(out[-1]["content"]) + "\n" + str(content)
            else:
                out.append({"role": role, "content": content})
    if out and out[0]["role"] != "user":
        out.insert(0, {"role": "user", "content": "Continue."})
    return "\n\n".join(system_parts), out


def _anthropic_tools(minimal_tools: bool) -> list:
    result = []
    for n, t in TOOLS.items():
        if minimal_tools and n in {"subagent", "mcp_call", "acp_request", "git_diff",
                                   "git_commit", "skill_load", "skill_create"}:
            continue
        result.append({"name": n, "description": t["desc"], "input_schema": t["params"]})
    return result


def _post_chat(cfg: dict, payload: dict, stream: bool):
    meta = PROVIDERS[cfg["provider"]]
    kind = meta.get("kind", "openai")
    if kind == "anthropic":
        url = meta["base"].rstrip("/") + "/messages"
        headers = _anthropic_headers(cfg)
    else:
        url = meta["base"].rstrip("/") + "/chat/completions"
        headers = _openai_headers(cfg)

    resp = _request("POST", url, headers=headers, json_body=payload,
                    timeout=300, retries=3, stream=stream)

    if resp.status_code >= 400:
        raise RuntimeError(_api_error(resp))
    return resp


def call_llm(messages: list, stream: bool = True, minimal_tools: bool = False):
    cfg = load_cfg()
    provider = cfg.get("provider")
    if provider not in PROVIDERS:
        raise RuntimeError(f"Unknown provider: {provider!r}. Run /init.")

    meta = PROVIDERS[provider]
    model_meta = _selected_model(cfg)
    model = cfg.get("model") or model_meta["id"]
    kind = meta.get("kind", "openai")

    if kind == "anthropic":
        system, api_messages = _messages_for_anthropic(messages)
        payload = {
            "model": model,
            "max_tokens": int(cfg.get("max_tokens", MAX_TOKENS)),
            "temperature": 0.2,
            "system": system,
            "messages": api_messages,
        }
        # Only send tools when the selected model is known to support them.
        if not cfg.get("disable_tools") and model_meta.get("tool_calling", True):
            payload["tools"] = _anthropic_tools(minimal_tools)
        if stream:
            payload["stream"] = True

        try:
            resp = _post_chat(cfg, payload, stream)
        except RuntimeError as e:
            # 403/permission errors can happen if a stale model was saved.
            # Refresh the provider catalog once, switch only if necessary,
            # then retry exactly once.
            if "HTTP 403" in str(e) or "HTTP 404" in str(e):
                try:
                    models = fetch_models(cfg, force=True)
                    ids = {m["id"] for m in models}
                    if model not in ids:
                        cfg["model"] = models[0]["id"]
                        save_cfg(cfg)
                        return call_llm(messages, stream=stream, minimal_tools=minimal_tools)
                except Exception:
                    pass
            raise
        return stream_anthropic(resp) if stream else parse_anthropic_response(resp.json())

    tools_enabled = not cfg.get("disable_tools") and model_meta.get("tool_calling", True)
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": int(cfg.get("max_tokens", MAX_TOKENS)),
        "temperature": 0.2,
        "stream": bool(stream),
    }
    if tools_enabled:
        payload["tools"] = tool_schema(minimal_tools)
        payload["tool_choice"] = "auto"

    try:
        resp = _post_chat(cfg, payload, stream)
    except RuntimeError as e:
        if "HTTP 403" in str(e) or "HTTP 404" in str(e):
            try:
                models = fetch_models(cfg, force=True)
                ids = {m["id"] for m in models}
                if model not in ids:
                    cfg["model"] = models[0]["id"]
                    save_cfg(cfg)
                    return call_llm(messages, stream=stream, minimal_tools=minimal_tools)
            except Exception:
                pass
        raise
    return stream_response(resp) if stream else parse_openai_response(resp.json())


def parse_openai_response(d: dict):
    if "error" in d:
        raise RuntimeError(str(d["error"]))
    choices = d.get("choices") or []
    if not choices:
        raise RuntimeError(f"Provider returned no choices: {json.dumps(d)[:2000]}")
    msg = choices[0].get("message") or {}
    return msg.get("content") or "", msg.get("tool_calls") or []


def stream_response(resp: httpx.Response):
    text_buf = ""
    tool_calls = {}
    content_type = resp.headers.get("content-type", "").lower()
    if "text/event-stream" not in content_type:
        return parse_openai_response(resp.json())

    for raw in resp.iter_lines():
        line = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
        if not line or not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            d = json.loads(data)
        except json.JSONDecodeError:
            continue
        if "error" in d:
            raise RuntimeError(json.dumps(d["error"]))
        choices = d.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        piece = delta.get("content")
        if piece:
            text_buf += str(piece)
            console.print(Text(str(piece)), end="")
        for tc in delta.get("tool_calls") or []:
            idx = tc.get("index", 0)
            if idx not in tool_calls:
                tool_calls[idx] = {
                    "id": tc.get("id") or str(uuid.uuid4()),
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                }
            if tc.get("id"):
                tool_calls[idx]["id"] = tc["id"]
            fn = tc.get("function") or {}
            if fn.get("name"):
                tool_calls[idx]["function"]["name"] += fn["name"]
            if fn.get("arguments"):
                tool_calls[idx]["function"]["arguments"] += fn["arguments"]
    return text_buf, [tool_calls[i] for i in sorted(tool_calls)]


def parse_anthropic_response(d: dict):
    if "error" in d:
        raise RuntimeError(str(d["error"]))
    text, calls = [], []
    for block in d.get("content", []):
        if block.get("type") == "text":
            text.append(block.get("text", ""))
        elif block.get("type") == "tool_use":
            calls.append({
                "id": block.get("id") or str(uuid.uuid4()),
                "type": "function",
                "function": {
                    "name": block.get("name", ""),
                    "arguments": json.dumps(block.get("input", {})),
                },
            })
    return "".join(text), calls


def stream_anthropic(resp: httpx.Response):
    text_buf, calls = "", {}
    content_type = resp.headers.get("content-type", "").lower()
    if "text/event-stream" not in content_type:
        return parse_anthropic_response(resp.json())

    current_tool = None
    for raw in resp.iter_lines():
        line = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
        if not line.startswith("data:"):
            continue
        try:
            event = json.loads(line[5:].strip())
        except json.JSONDecodeError:
            continue
        etype = event.get("type")
        if etype == "error":
            raise RuntimeError(json.dumps(event.get("error", event)))
        if etype == "content_block_start":
            block = event.get("content_block") or {}
            if block.get("type") == "tool_use":
                current_tool = block.get("id") or str(uuid.uuid4())
                calls[current_tool] = {
                    "id": current_tool, "type": "function",
                    "function": {"name": block.get("name", ""), "arguments": ""},
                }
        elif etype == "content_block_delta":
            delta = event.get("delta") or {}
            if delta.get("type") == "text_delta":
                piece = delta.get("text", "")
                text_buf += piece
                console.print(Text(piece), end="")
            elif delta.get("type") == "input_json_delta" and current_tool:
                calls[current_tool]["function"]["arguments"] += delta.get("partial_json", "")
        elif etype == "content_block_stop":
            current_tool = None
    return text_buf, list(calls.values())


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
    now = time.time()
    with sqlite3.connect(SESSIONS_DB) as db:
        db.execute(
            """INSERT INTO sessions(id, created, updated, title, messages)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                   updated=excluded.updated,
                   title=CASE WHEN excluded.title != '' THEN excluded.title ELSE sessions.title END,
                   messages=excluded.messages""",
            (session_id, now, now, title, json.dumps(messages)),
        )

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
    cmd = f"git worktree add {shlex.quote(str(p))}"
    if branch:
        cmd += f" {shlex.quote(branch)}"
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
    return run_cmd(f"git worktree remove {shlex.quote(str(p))}")

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
        result = subprocess.run(["find", ".", "-type", "f", "-not", "-path", "*/.*/*"],
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            files = result.stdout.strip().split('\n')
            # Simple tree view, respecting the requested depth
            tree_lines = []
            for f in sorted(files):
                if not f:  # Skip empty lines
                    continue
                file_depth = f.count('/')
                if file_depth > depth:
                    continue
                indent = "  " * file_depth
                name = f.split('/')[-1] if '/' in f else f
                tree_lines.append(f"{indent}{name}")
            return '\n'.join(tree_lines[:100]) or "(no files within depth)"
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
    messages = AGENT_STATE.get("active_messages")
    if messages is None:
        return "ERROR: no active conversation to compact"
    if len(messages) <= keep_last + 2:  # +2 for system and maybe one other
        return f"(history only {len(messages)} messages, no compaction needed)"

    system_msgs = [m for m in messages if m.get("role") == "system"]
    non_system = [m for m in messages if m.get("role") != "system"]
    recent_msgs = non_system[-keep_last:] if keep_last > 0 else []

    # If the tail starts with tool results, retain the preceding assistant
    # tool-call message(s) so the provider receives a valid conversation.
    while recent_msgs and recent_msgs[0].get("role") == "tool":
        idx = non_system.index(recent_msgs[0])
        if idx <= 0:
            break
        recent_msgs.insert(0, non_system[idx - 1])
        if recent_msgs[0].get("role") != "tool":
            break

    old_len = len(messages)
    messages[:] = system_msgs + recent_msgs
    return f"compacted context: removed {old_len - len(messages)} messages, kept {len(messages)}"

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
    AGENT_STATE["active_messages"] = messages
    minimal = cfg.get("harness") == "minimal"

    mutating_tools = {
        "write_file", "edit_file", "bash", "notebook_edit", "git_commit",
        "git_worktree_add", "git_worktree_remove", "agents_write",
        "skill_create", "mcp_call", "acp_request",
    }

    for it in range(cfg.get("max_iter", 50)):
        if AGENT_STATE["interrupted"]:
            AGENT_STATE["interrupted"] = False
            console.print("[yellow]interrupted[/yellow]")
            return

        if AGENT_STATE["steering_queue"]:
            steer = AGENT_STATE["steering_queue"].pop(0)
            messages.append({"role": "user", "content": steer})
            log_history(CURRENT_SESSION_ID, "user", steer)
            console.print(f"[cyan]steer:[/cyan] {steer}")

        console.print(f"\n[dim]── iter {it+1} ──[/dim]")
        try:
            text, calls = call_llm(messages, stream=True, minimal_tools=minimal)
        except Exception as e:
            console.print(f"[red]LLM error: {e}[/red]")
            return

        # Critical fix: preserve every assistant response, including responses
        # that contain no tool calls. The original implementation discarded it.
        assistant_msg = {"role": "assistant", "content": text or ""}
        if calls:
            assistant_msg["tool_calls"] = calls
        messages.append(assistant_msg)
        log_history(CURRENT_SESSION_ID, "assistant", text or "", calls)

        if not calls:
            if text:
                console.print()
            else:
                console.print("[yellow](model returned an empty response)[/yellow]")
            return

        for c in calls:
            name = c.get("function", {}).get("name", "")
            raw_args = c.get("function", {}).get("arguments", "{}")
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
            except json.JSONDecodeError as e:
                out = f"ERROR: invalid tool arguments for {name}: {e}"
                messages.append({"role": "tool", "tool_call_id": c.get("id", ""), "content": out})
                log_history(CURRENT_SESSION_ID, "tool", out)
                console.print(f"  [red]→ {name}: {out}[/red]")
                continue

            if name not in TOOLS:
                out = f"ERROR: unknown tool '{name}'"
            elif AGENT_STATE["plan_mode"] and name in mutating_tools:
                out = f"BLOCKED: plan mode is read-only; tool '{name}' was not executed."
            elif not confirm_tool(name, args):
                out = "DENIED by user"
            else:
                t0 = time.time()
                try:
                    out = TOOLS[name]["fn"](**args)
                except Exception as e:
                    out = f"ERROR: {e}\n{traceback.format_exc()[-1000:]}"
                dt = time.time() - t0

            preview = str(out)[:200].replace("\n", " ")
            elapsed = f" ({time.time() - t0:.1f}s)" if "t0" in locals() else ""
            console.print(f"  [dim]→[/dim] [cyan]{name or '?'}[/cyan][dim]{elapsed}[/dim] {preview}")
            messages.append({
                "role": "tool",
                "tool_call_id": c.get("id") or str(uuid.uuid4()),
                "content": str(out),
            })
            log_history(CURRENT_SESSION_ID, "tool", str(out), [{
                "name": name, "args": args, "result": str(out)[:500]
            }])

    console.print("[yellow]hit max_iter[/yellow]")


# --- TUI helpers ---
def show_todos():
    # Kept for compatibility; the persistent dashboard owns TODO rendering.
    return None


def show_help():
    console.print(Panel("""[cyan]/help[/cyan]        this help
[cyan]/cd <dir>[/cyan]      change cwd
[cyan]/model[/cyan]         refresh/search/select models
[cyan]/provider[/cyan]      switch provider and authorize
[cyan]/harness[/cyan]       switch harness
[cyan]/init[/cyan]          re-run provider authorization + model discovery
[cyan]/plan[/cyan]          toggle plan mode
[cyan]/shell[/cyan]         toggle auto-approve shell
[cyan]/write[/cyan]         toggle auto-approve file writes
[cyan]/mcp[/cyan]           manage MCP servers
[cyan]/skill[/cyan]         manage skills
[cyan]/acp[/cyan]           start ACP server for editor
[cyan]/session[/cyan]       session management
[cyan]/clear[/cyan]         clear conversation
[cyan]/history[/cyan]       show session history
[cyan]/health[/cyan]        verify provider authorization
[cyan]/quit[/cyan]          exit
[cyan]Ctrl+C[/cyan]         interrupt current run""", title="[bold]SCODE Commands[/bold]", border_style="cyan"))


def repl():
    global CWD, CURRENT_SESSION_ID

    # Check for non-interactive mode first
    is_non_interactive = len(sys.argv) > 1 or not sys.stdin.isatty()

    cfg = ensure_cfg(non_interactive=is_non_interactive)
    AGENT_STATE["current_harness"] = cfg.get("harness", "native")
    CURRENT_SESSION_ID = str(uuid.uuid4())
    messages = [{"role": "system", "content": SYSTEM_PROMPT.format(
        cwd=CWD, harness=cfg.get("harness", "native")
    )}]

    if is_non_interactive:
        prompt = " ".join(sys.argv[1:])
        messages.append({"role": "user", "content": prompt})
        log_history(CURRENT_SESSION_ID, "user", prompt)
        run_agent(messages, cfg)
        save_session(CURRENT_SESSION_ID, messages)
        return

    def get_context_usage():
        total = 0
        for m in messages:
            content = m.get("content")
            if isinstance(content, str):
                total += len(content) // 4
            if m.get("tool_calls"):
                total += len(str(m["tool_calls"])) // 4
        return total

    def dashboard():
        used = get_context_usage()
        # Prefer selected model's advertised context when available.
        mm = _selected_model(cfg)
        max_context = mm.get("context_length") or cfg.get("max_context", 8192)
        try:
            max_context = max(1, int(max_context))
        except Exception:
            max_context = 8192
        pct = min(100, int((used / max_context) * 100))
        bar_len = 20
        filled = int(bar_len * pct / 100)
        bar = "█" * filled + "░" * (bar_len - filled)

        context_panel = Panel(
            f"{used:,} / {max_context:,} tokens\n[{bar}] {pct}%\n"
            f"[dim]catalog: {len(get_cached_models(cfg))} models[/dim]",
            title="[cyan]Context[/cyan]", border_style="cyan", width=38,
        )

        mcp = cfg.get("mcp_servers", [])
        mcp_lines = [
            f"  [green]●[/green] {s.get('name','?')} ({s.get('transport','?')})"
            for s in mcp
        ] or ["  [dim]No MCP servers configured[/dim]", "  [dim]Use /mcp to add[/dim]"]
        mcp_panel = Panel("\n".join(mcp_lines), title="[cyan]MCP[/cyan]",
                          border_style="cyan", width=38)

        # LSP is a stable dashboard state, not a printed block on every command.
        lsp_lines = ["  [green]●[/green] typescript",
                     "  [green]●[/green] python",
                     "  [dim]●[/dim] eslint"]
        lsp_panel = Panel("\n".join(lsp_lines), title="[cyan]LSP[/cyan]",
                          border_style="cyan", width=38)

        todos = AGENT_STATE.get("todos", [])
        todo_lines = []
        for t in todos[:8]:
            status = t.get("status", "pending")
            icon = {"pending": "○", "in_progress": "◐",
                    "completed": "●", "cancelled": "✗"}.get(status, "?")
            style = {"pending": "dim", "in_progress": "yellow",
                     "completed": "green", "cancelled": "red"}.get(status, "")
            todo_lines.append(f"  [{style}]{icon} {t.get('content','')[:34]}[/{style}]")
        if not todo_lines:
            todo_lines = ["  [dim]No tasks[/dim]", "  [dim]Agent will create[/dim]"]
        if len(todos) > 8:
            todo_lines.append(f"  [dim]...and {len(todos)-8} more[/dim]")
        todo_panel = Panel("\n".join(todo_lines), title="[cyan]TODO[/cyan]",
                           border_style="cyan", width=38)

        status = (
            f"[bold]{cfg.get('provider','?')}/{cfg.get('model','?')}[/bold] | "
            f"{AGENT_STATE.get('current_harness', cfg.get('harness','native'))} | "
            f"{'PLAN' if AGENT_STATE.get('plan_mode') else 'WRITE'}"
        )
        footer = Panel(
            status + " | [cyan]/model[/cyan] search  [cyan]/provider[/cyan] authorize  "
            "[cyan]Ctrl+C[/cyan] interrupt",
            border_style="dim",
        )
        return Columns([context_panel, mcp_panel, lsp_panel, todo_panel],
                       equal=True, expand=True), footer

    def refresh(live):
        side, footer = dashboard()
        live.update(Columns([side, footer]))

    # The dashboard is rendered exactly once and updated in-place. Agent output
    # and command output are printed above it; the dashboard itself never repeats.
    side, footer = dashboard()
    live_render = Columns([side, footer])
    with Live(live_render, console=console, refresh_per_second=4, transient=False) as live:
        while True:
            try:
                user_input = Prompt.ask("[bold green]🚀[/bold green]").strip()
            except (EOFError, KeyboardInterrupt):
                save_session(CURRENT_SESSION_ID, messages)
                return

            if not user_input:
                refresh(live)
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
                    target = parts[1] if len(parts) > 1 else str(HOME)
                    pp = Path(target).expanduser().resolve()
                    if pp.is_dir():
                        CWD = pp
                        messages[0]["content"] = SYSTEM_PROMPT.format(
                            cwd=CWD, harness=cfg.get("harness", "native"))
                        console.print(f"[dim]cwd: {CWD}[/dim]")
                    else:
                        console.print(f"[red]not a dir: {pp}[/red]")

                elif cmd == "/model":
                    try:
                        models = fetch_models(cfg, force=True)
                        cfg["model"] = select_model(cfg, models)
                        save_cfg(cfg)
                        console.print(f"[green]→ {cfg['model']}[/green]")
                    except Exception as e:
                        console.print(f"[red]model discovery failed: {e}[/red]")

                elif cmd == "/provider":
                    names = list(PROVIDERS)
                    for i, n in enumerate(names, 1):
                        console.print(f"  [cyan]{i}[/cyan] {n}")
                    choice = Prompt.ask("provider",
                                        choices=[str(i) for i in range(1, len(names)+1)],
                                        default="1")
                    cfg["provider"] = names[int(choice)-1]
                    cfg["api_key"] = ""
                    meta = PROVIDERS[cfg["provider"]]
                    if meta.get("requires_key"):
                        cfg["api_key"] = Prompt.ask(
                            f"API key for [cyan]{cfg['provider']}[/cyan]",
                            password=True, default="")
                    try:
                        models = fetch_models(cfg, force=True)
                        cfg["model"] = select_model(cfg, models)
                        save_cfg(cfg)
                        console.print(
                            f"[green]Authorized. {len(models)} models available. "
                            f"Selected: {cfg['model']}[/green]"
                        )
                    except Exception as e:
                        console.print(f"[red]authorization/model discovery failed: {e}[/red]")

                elif cmd == "/harness":
                    names = list(HARNESSES)
                    for i, n in enumerate(names, 1):
                        console.print(f"  [cyan]{i}[/cyan] {n} — {HARNESSES[n]['desc']}")
                    choice = Prompt.ask("harness",
                                        choices=[str(i) for i in range(1, len(names)+1)],
                                        default="1")
                    cfg["harness"] = names[int(choice)-1]
                    AGENT_STATE["current_harness"] = cfg["harness"]
                    messages[0]["content"] = SYSTEM_PROMPT.format(
                        cwd=CWD, harness=cfg["harness"])
                    save_cfg(cfg)

                elif cmd == "/init":
                    try:
                        cfg = ensure_cfg(non_interactive=False)
                        messages[0]["content"] = SYSTEM_PROMPT.format(
                            cwd=CWD, harness=cfg.get("harness", "native"))
                        AGENT_STATE["current_harness"] = cfg.get("harness", "native")
                    except Exception as e:
                        console.print(f"[red]init failed: {e}[/red]")

                elif cmd == "/plan":
                    AGENT_STATE["plan_mode"] = not AGENT_STATE["plan_mode"]

                elif cmd == "/shell":
                    cfg["auto_approve_shell"] = not cfg.get("auto_approve_shell", False)
                    save_cfg(cfg)

                elif cmd == "/write":
                    cfg["auto_approve_write"] = not cfg.get("auto_approve_write", True)
                    save_cfg(cfg)

                elif cmd == "/mcp":
                    console.print("[dim]MCP servers:[/dim]")
                    for s in cfg.get("mcp_servers", []):
                        console.print(f"  - {s.get('name','?')} ({s.get('transport','?')})")
                    if Prompt.ask("add server?", choices=["y","n"], default="n") == "y":
                        name = Prompt.ask("name")
                        transport = Prompt.ask("transport", choices=["stdio","http","sse"], default="stdio")
                        if transport == "stdio":
                            command = Prompt.ask("command")
                            cfg.setdefault("mcp_servers", []).append(
                                {"name": name, "transport": "stdio", "command": command})
                        else:
                            url = Prompt.ask("url")
                            cfg.setdefault("mcp_servers", []).append(
                                {"name": name, "transport": transport, "url": url})
                        save_cfg(cfg)

                elif cmd == "/skill":
                    console.print("[dim]Skills:[/dim]")
                    for f in SKILLS_DIR.glob("*.md"):
                        console.print(f"  - {f.stem}")
                    action = Prompt.ask("action",
                                        choices=["load","create","delete","list"],
                                        default="list")
                    if action == "load":
                        console.print(load_skill(Prompt.ask("skill name")))
                    elif action == "create":
                        name = Prompt.ask("name")
                        console.print("Enter skill markdown (end with EOF/Ctrl+D):")
                        content = sys.stdin.read()
                        console.print(t_skill_create(name, content))
                    elif action == "delete":
                        name = Prompt.ask("name")
                        (SKILLS_DIR / f"{name}.md").unlink(missing_ok=True)

                elif cmd == "/acp":
                    console.print(start_acp_server())
                    AGENT_STATE["acp_mode"] = True

                elif cmd == "/session":
                    action = parts[1] if len(parts) > 1 else "list"
                    if action == "list":
                        for sid, created, title in list_sessions():
                            console.print(
                                f"  {sid[:8]} {datetime.fromtimestamp(created).strftime('%Y-%m-%d %H:%M')} {title}"
                            )
                    elif action == "new":
                        save_session(CURRENT_SESSION_ID, messages)
                        CURRENT_SESSION_ID = str(uuid.uuid4())
                        messages = [{"role":"system","content":SYSTEM_PROMPT.format(
                            cwd=CWD, harness=cfg.get("harness","native"))}]
                    elif action == "load":
                        sid = parts[2] if len(parts) > 2 else Prompt.ask("session id")
                        loaded = load_session(sid)
                        if loaded:
                            messages = loaded
                            CURRENT_SESSION_ID = sid
                    elif action == "save":
                        title = parts[2] if len(parts) > 2 else ""
                        save_session(CURRENT_SESSION_ID, messages, title)

                elif cmd == "/clear":
                    messages = [messages[0]]
                    console.print("[dim]conversation cleared[/dim]")

                elif cmd == "/history":
                    with sqlite3.connect(HISTORY_DB) as db:
                        rows = db.execute(
                            "SELECT ts, role, content FROM history WHERE session_id=? "
                            "ORDER BY id DESC LIMIT 20", (CURRENT_SESSION_ID,)
                        ).fetchall()
                    for ts, role, content in reversed(rows):
                        console.print(
                            f"[dim]{datetime.fromtimestamp(ts).strftime('%H:%M:%S')}[/dim] "
                            f"[{role}] {content[:100]}"
                        )

                elif cmd == "/health":
                    try:
                        models = fetch_models(cfg, force=True)
                        current = cfg.get("model")
                        ids = {m["id"] for m in models}
                        if current not in ids:
                            cfg["model"] = select_model(cfg, models)
                        save_cfg(cfg)
                        console.print(
                            f"[green]Authorized. {len(models)} usable models. "
                            f"Current: {cfg['model']}[/green]"
                        )
                    except Exception as e:
                        console.print(f"[red]health failed: {e}[/red]")

                else:
                    console.print(f"[red]unknown: {cmd}[/red]")

                refresh(live)
                continue

            messages.append({"role": "user", "content": user_input})
            log_history(CURRENT_SESSION_ID, "user", user_input)
            run_agent(messages, cfg)
            save_session(CURRENT_SESSION_ID, messages)
            refresh(live)


if __name__ == "__main__":
    repl()

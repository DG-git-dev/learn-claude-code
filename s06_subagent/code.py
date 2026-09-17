#!/usr/bin/env python3
"""
s06_subagent.py - Subagents

The task tool runs a second agent loop with a fresh input list. Both
loops share the working directory, but only the final text returns to
the parent conversation.

    Parent agent                    Subagent
    +------------------+            +------------------+
    | input=[...]      |            | input=[prompt]   |
    |                  |   task     |                  |
    | tool: task       | ---------> | own agent loop   |
    |                  |            | base tools only  |
    | tool_result      | <--------- | final text       |
    +------------------+            +------------------+

The subagent has no task tool, so it cannot delegate again.
"""

import json
import os
import re
import subprocess
from pathlib import Path

try:
    import readline
    readline.parse_and_bind('set bind-tty-special-chars off')
except ImportError:
    pass

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

WORKDIR = Path.cwd()
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "copilot-bridge"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)
MODEL = os.environ["MODEL_ID"]

SYSTEM = (
    f"You are a coding agent at {WORKDIR}. "
    "Use task for focused exploration or a self-contained subtask."
)
SUB_SYSTEM = (
    f"You are a coding agent at {WORKDIR}. "
    "Complete the given task, then return a concise final answer."
)


# -- Base tools --

def run_bash(command: str) -> str:
    """Run one shell command and return bounded combined output."""
    try:
        result = subprocess.run(
            command, shell=True, cwd=WORKDIR,
            capture_output=True, text=True, errors="replace", timeout=120,
        )
        output = (result.stdout + result.stderr).strip()
        return output[:50000] if output else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


def run_read(path: str, limit: int | None = None) -> str:
    """Read a UTF-8 file, optionally limiting the returned line count."""
    try:
        lines = (WORKDIR / path).resolve().read_text(encoding="utf-8").splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    """Write UTF-8 content to a file, creating parent directories."""
    try:
        file_path = (WORKDIR / path).resolve()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    """Replace the first exact text match in a UTF-8 file."""
    try:
        file_path = (WORKDIR / path).resolve()
        text = file_path.read_text(encoding="utf-8")
        if old_text not in text:
            return f"Error: text not found in {path}"
        file_path.write_text(text.replace(old_text, new_text, 1), encoding="utf-8")
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


def run_glob(pattern: str) -> str:
    """Return up to 200 workspace files matching a glob pattern."""
    import glob as g
    try:
        matches = sorted({
            match for match in g.glob(
                pattern, root_dir=WORKDIR, recursive=True)
            if (WORKDIR / match).resolve().is_relative_to(WORKDIR)
        })
        shown = matches[:200]
        if len(matches) > 200:
            shown.append("... (more matches omitted; narrow the pattern)")
        return "\n".join(shown) if shown else "(no matches)"
    except Exception as e:
        return f"Error: {e}"


BASE_TOOLS = [
    {"type": "function", "name": "bash", "description": "Run a shell command.",
     "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"], "additionalProperties": False}},
    {"type": "function", "name": "read_file", "description": "Read file contents.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"], "additionalProperties": False}},
    {"type": "function", "name": "write_file", "description": "Write content to a file.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"], "additionalProperties": False}},
    {"type": "function", "name": "edit_file", "description": "Replace exact text in a file once.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"], "additionalProperties": False}},
    {"type": "function", "name": "glob", "description": "Find files matching a glob pattern; ** matches recursively.",
     "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"], "additionalProperties": False}},
]

BASE_HANDLERS = {
    "bash": run_bash,
    "read_file": run_read,
    "write_file": run_write,
    "edit_file": run_edit,
    "glob": run_glob,
}


# -- Hooks --

HOOKS = {"UserPromptSubmit": [], "PreToolUse": [], "PostToolUse": [], "Stop": []}


def register_hook(event: str, callback):
    """Append a callback to an event's ordered hook list."""
    HOOKS[event].append(callback)


def trigger_hooks(event: str, *args):
    """Run callbacks in order and return the first blocking result."""
    for callback in HOOKS[event]:
        result = callback(*args)
        if result is not None:
            return result
    return None


DENY_LIST = ["rm -rf /", "sudo", "shutdown", "reboot", "mkfs", "dd if="]
DESTRUCTIVE_COMMAND_WORD = re.compile(
    r"(?i)(?:^|[;&|()\n])\s*(?:rm|del)(?=\s|$|[;&|()])"
)
DESTRUCTIVE = ["rm ", "> /etc/", "chmod 777"]


def contains_destructive_command(command: str) -> bool:
    """Detect rm or del when used as an actual shell command word."""
    return bool(DESTRUCTIVE_COMMAND_WORD.search(command))


def permission_hook(tool_name: str, arguments: dict):
    """PreToolUse: block denied operations and ask about risky ones."""
    if tool_name == "bash":
        command = arguments.get("command", "")
        for pattern in DENY_LIST:
            if pattern in command:
                print(f"\n\033[31m[blocked] '{pattern}'\033[0m")
                return "Permission denied by deny list"
        if contains_destructive_command(command) or any(
            keyword in command for keyword in DESTRUCTIVE
        ):
            print("\n\033[33m[permission] Potentially destructive command\033[0m")
            print(f"   Tool: {tool_name}({arguments})")
            choice = input("   Allow? [y/N] ").strip().lower()
            if choice not in ("y", "yes"):
                return "Permission denied by user"

    if tool_name in ("read_file", "write_file", "edit_file"):
        path = arguments.get("path", "")
        if not (WORKDIR / path).resolve().is_relative_to(WORKDIR):
            print("\n\033[33m[permission] Access outside workspace\033[0m")
            print(f"   Tool: {tool_name}({arguments})")
            choice = input("   Allow? [y/N] ").strip().lower()
            if choice not in ("y", "yes"):
                return "Permission denied by user"
    return None


def log_hook(tool_name: str, arguments: dict):
    """PreToolUse: log every tool call."""
    args_preview = str(list(arguments.values())[:2])[:60]
    print(f"\033[90m[HOOK] {tool_name}({args_preview})\033[0m")
    return None


def large_output_hook(tool_name: str, arguments: dict, output):
    """PostToolUse: warn on large output."""
    if len(str(output)) > 100000:
        print(f"\033[33m[HOOK] Large output from {tool_name}: {len(str(output))} chars\033[0m")
    return None


def context_inject_hook(query: str):
    """UserPromptSubmit: log the working directory."""
    print(f"\033[90m[HOOK] UserPromptSubmit: working in {WORKDIR}\033[0m")
    return None


def summary_hook(items: list):
    """Stop: print the number of function results in this input list."""
    # Input dictionaries and typed SDK items coexist in a Responses input list.
    tool_count = sum(
        1 for item in items
        if (item.get("type") if isinstance(item, dict)
            else getattr(item, "type", None)) == "function_call_output"
    )
    print(f"\033[90m[HOOK] Stop: session used {tool_count} tool calls\033[0m")
    return None


register_hook("UserPromptSubmit", context_inject_hook)
register_hook("PreToolUse", permission_hook)
register_hook("PreToolUse", log_hook)
register_hook("PostToolUse", large_output_hook)
register_hook("Stop", summary_hook)


def execute_tool(call, handlers: dict) -> str:
    """Decode and execute one Responses function call through shared hooks."""
    # Responses function arguments arrive as serialized JSON text.
    arguments = json.loads(call.arguments)
    blocked = trigger_hooks("PreToolUse", call.name, arguments)
    if blocked:
        return str(blocked)

    handler = handlers.get(call.name)
    try:
        output = handler(**arguments) if handler else f"Unknown: {call.name}"
    except Exception as e:
        output = f"Error: {e}"

    trigger_hooks("PostToolUse", call.name, arguments, output)
    return str(output)


# -- New in s06: a nested agent loop with fresh input --

# A child gets the kernel tools but cannot recursively delegate.
SUB_TOOLS = list(BASE_TOOLS)
SUB_HANDLERS = dict(BASE_HANDLERS)


def run_subagent(prompt: str) -> str:
    """Run a fresh child context and return only its final text."""
    print("\n\033[35m[Subagent started]\033[0m")
    # Parent history is intentionally absent from this new input list.
    items = [{"role": "user", "content": prompt}]

    for _ in range(30):
        response = client.responses.create(
            model=MODEL,
            instructions=SUB_SYSTEM,
            input=items,
            tools=SUB_TOOLS,
            max_output_tokens=8000,
        )
        items.extend(response.output)

        tool_calls = [
            item for item in response.output if item.type == "function_call"
        ]
        if not tool_calls:
            force = trigger_hooks("Stop", items)
            if force:
                items.append({"role": "user", "content": force})
                continue
            print("\033[35m[Subagent done]\033[0m")
            # Only this summary crosses back into the parent's context.
            return response.output_text or "(no summary)"

        results = []
        for call in tool_calls:
            output = execute_tool(call, SUB_HANDLERS)
            print(f"  \033[90m[sub] {call.name}: {output[:100]}\033[0m")
            results.append({
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": output,
            })
        items.extend(results)

    print("\033[35m[Subagent stopped]\033[0m")
    return "Subagent stopped after 30 turns without a final answer."


TASK_TOOL = {
    "type": "function",
    "name": "task",
    "description": "Run a subagent with fresh conversation context and return its final text.",
    "parameters": {
        "type": "object",
        "properties": {"prompt": {"type": "string", "minLength": 1}},
        "required": ["prompt"],
        "additionalProperties": False,
    },
}

TOOLS = [*BASE_TOOLS, TASK_TOOL]
TOOL_HANDLERS = {**BASE_HANDLERS, "task": run_subagent}


# -- Parent agent loop --

def agent_loop(items: list) -> str:
    """Run the parent loop and return its final response text."""
    while True:
        response = client.responses.create(
            model=MODEL,
            instructions=SYSTEM,
            input=items,
            tools=TOOLS,
            max_output_tokens=8000,
        )
        items.extend(response.output)

        tool_calls = [
            item for item in response.output if item.type == "function_call"
        ]
        if not tool_calls:
            force = trigger_hooks("Stop", items)
            if force:
                items.append({"role": "user", "content": force})
                continue
            return response.output_text

        results = []
        for call in tool_calls:
            output = execute_tool(call, TOOL_HANDLERS)
            results.append({
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": output,
            })
        items.extend(results)


if __name__ == "__main__":
    print("s06: Subagent - fresh messages, final text returns")
    print("Enter a question, press Enter to send. Type q to quit.\n")

    history = []
    while True:
        try:
            # \001/\002 tell Readline the ANSI escapes have zero display width.
            query = input("\001\033[36m\002s06 >> \001\033[0m\002")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        trigger_hooks("UserPromptSubmit", query)
        history.append({"role": "user", "content": query})
        print(agent_loop(history))
        print()

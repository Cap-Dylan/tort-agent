#!/usr/bin/env python3
"""
Tort Agent — Local AI assistant powered by Ollama.
Phase 2: work/play mode toggle + tool calling.
"""

import json
import sys
import requests

from rich.console import Console
from rich.panel import Panel

from config import OLLAMA_URL, MODEL, SYSTEM_PROMPT, MAX_HISTORY
from tools import TOOL_REGISTRY, execute_tool


# ── Setup ────────────────────────────────────────────────────────────────────

console = Console()

# Conversation history — system prompt is injected per-request, not stored here.
history: list[dict] = []

# Mode state. "play" = persona chat (streamed). "work" = lean tool-using mode.
mode = "play"
think_next = False

# Lean system prompt for work mode. No persona — just clear instructions.
# The reasoning distillation will think before answering regardless;
# this prompt just keeps the *output* tight and tool-focused.
WORK_SYSTEM_PROMPT = """You are Tort Agent in work mode — Dylan's local study assistant.

You have tools that operate on Dylan's Obsidian vault, Canvas LMS, and handwritten
notes. When a request needs information you don't have, call a tool. When the tool
result answers the request, summarize it concisely and stop. Don't call tools you
don't need. Don't roleplay. Be direct.
"""

# Hard cap on tool-call iterations per turn — stops runaway loops if the model
# keeps calling the same tool. Bump this if a routine legitimately needs more.
MAX_TOOL_ITERATIONS = 5


# ── Tool registry adapter ────────────────────────────────────────────────────

def _ollama_tools() -> list[dict]:
    """
    Convert flat TOOL_REGISTRY entries to the shape Ollama's /api/chat expects:
        {"type": "function", "function": {name, description, parameters}}
    Keeps tools.py readable while satisfying the API.
    """
    return [{"type": "function", "function": entry} for entry in TOOL_REGISTRY]


# ── Streaming chat (play mode) ───────────────────────────────────────────────

def chat_play(user_message: str) -> str:
    """Persona chat with streaming. Strips <think> blocks for display."""
    history.append({"role": "user", "content": user_message})
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

    global think_next
    use_thinking = think_next
    think_next = False

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": MODEL,
                "messages": messages,
                "stream": True,
                "think": use_thinking,
            },
            stream=True,
            timeout=120,
        )
        response.raise_for_status()
    except requests.ConnectionError:
        console.print("[red]Can't reach Ollama. Is 'ollama serve' running?[/red]")
        history.pop()
        return ""
    except requests.Timeout:
        console.print("[red]Ollama timed out.[/red]")
        history.pop()
        return ""

    full_response = []
    in_thinking = False
    thinking_buffer = ""

    # Only stall the print if the buffer literally ends with the start of "<think".
    # The old `"<" in buf and not ">" in buf` check stalled on any "<", which
    # broke Python code, comparisons, HTML, etc.
    THINK_PREFIXES = ("<", "<t", "<th", "<thi", "<thin", "<think")

    for line in response.iter_lines():
        if not line:
            continue
        try:
            chunk = json.loads(line)
        except json.JSONDecodeError:
            continue

        if chunk.get("done", False):
            break

        token = chunk.get("message", {}).get("content", "")
        if not token:
            continue

        thinking_buffer += token

        # Entering <think>
        if "<think>" in thinking_buffer and not in_thinking:
            in_thinking = True
            before = thinking_buffer.split("<think>")[0]
            if before:
                console.print(before, end="")
                full_response.append(before)
            console.print("\n[dim]─── thinking ───[/dim]")
            thinking_buffer = ""
            continue

        # Exiting </think>
        if "</think>" in thinking_buffer and in_thinking:
            in_thinking = False
            console.print("\n[dim]─── done thinking ───[/dim]\n")
            thinking_buffer = thinking_buffer.split("</think>", 1)[-1]
            continue

        if in_thinking:
            console.print(f"[dim]{thinking_buffer}[/dim]", end="")
            thinking_buffer = ""
            continue

        # Don't print yet if we might be mid-tag
        if any(thinking_buffer.endswith(p) for p in THINK_PREFIXES):
            continue

        console.print(thinking_buffer, end="")
        full_response.append(thinking_buffer)
        thinking_buffer = ""

    if thinking_buffer and not in_thinking:
        console.print(thinking_buffer, end="")
        full_response.append(thinking_buffer)

    console.print()

    assistant_message = "".join(full_response).strip()
    if assistant_message:
        history.append({"role": "assistant", "content": assistant_message})

    while len(history) > MAX_HISTORY:
        history.pop(0)

    return assistant_message


# ── Tool-calling chat (work mode) ─────────────────────────────────────────────

def chat_work(user_message: str) -> str:
    """
    Tool-using mode. Non-streaming — batch responses are far more reliable
    when tool_calls are involved.

    Loop:
      1. Send messages + tools to Ollama.
      2. If response has tool_calls: execute each, append results, loop.
      3. Otherwise: that's the final answer. Print and return.
    """
    history.append({"role": "user", "content": user_message})

    # Working copy — intermediate tool exchanges don't pollute long-term history.
    messages = [{"role": "system", "content": WORK_SYSTEM_PROMPT}] + list(history)
    tools = _ollama_tools()
    final_text = ""

    for iteration in range(MAX_TOOL_ITERATIONS):
        try:
            r = requests.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": MODEL,
                    "messages": messages,
                    "tools": tools,
                    "stream": False,
                    # This model always thinks per its training. Let it.
                    "think": True,
                },
                timeout=180,
            )
            r.raise_for_status()
        except requests.ConnectionError:
            console.print("[red]Can't reach Ollama.[/red]")
            history.pop()
            return ""
        except requests.Timeout:
            console.print("[red]Ollama timed out.[/red]")
            history.pop()
            return ""

        data = r.json()
        msg = data.get("message", {})
        tool_calls = msg.get("tool_calls", [])
        content = msg.get("content", "") or ""

        # Feed the assistant turn back so the next iteration sees what it just said.
        messages.append(msg)

        if tool_calls:
            console.print(f"[dim]→ {len(tool_calls)} tool call(s)[/dim]")
            for tc in tool_calls:
                fn = tc.get("function", {})
                name = fn.get("name", "")
                args = fn.get("arguments", {})

                # Ollama sometimes returns arguments as a JSON string instead of dict.
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}

                console.print(f"[dim]  {name}({args})[/dim]")
                result = execute_tool(name, args)
                preview = result[:120] + ("..." if len(result) > 120 else "")
                console.print(f"[dim]  ← {preview}[/dim]")

                messages.append({
                    "role": "tool",
                    "name": name,
                    "content": result,
                })
            continue  # Loop so the model can react to tool results

        # No tool_calls → final answer. Strip any leading <think> block for display.
        final_text = content.strip()
        if "<think>" in final_text and "</think>" in final_text:
            final_text = final_text.split("</think>", 1)[-1].strip()
        break
    else:
        final_text = "[Hit max tool iterations — stopping. Bump MAX_TOOL_ITERATIONS if this was legit.]"

    if final_text:
        console.print(final_text)
        history.append({"role": "assistant", "content": final_text})

    while len(history) > MAX_HISTORY:
        history.pop(0)

    return final_text


# ── Dispatch ──────────────────────────────────────────────────────────────────

def chat(user_message: str) -> str:
    return chat_work(user_message) if mode == "work" else chat_play(user_message)


# ── Commands ──────────────────────────────────────────────────────────────────

def handle_command(cmd: str) -> bool:
    global mode, think_next
    cmd_lower = cmd.strip().lower()

    if cmd_lower in ("/quit", "/exit", "/q"):
        console.print("\n[dim]See you later, Tort.[/dim]")
        sys.exit(0)

    elif cmd_lower == "/clear":
        history.clear()
        console.print("[dim]Conversation cleared.[/dim]")
        return True

    elif cmd_lower == "/model":
        console.print(f"[dim]Model: {MODEL}[/dim]")
        console.print(f"[dim]Endpoint: {OLLAMA_URL}[/dim]")
        console.print(f"[dim]Mode: {mode}[/dim]")
        return True

    elif cmd_lower == "/work":
        mode = "work"
        console.print("[dim]Work mode — tools enabled, persona off.[/dim]")
        return True

    elif cmd_lower == "/play":
        mode = "play"
        console.print("[dim]Play mode — persona on, tools off.[/dim]")
        return True

    elif cmd_lower == "/tools":
        names = [t["name"] for t in TOOL_REGISTRY]
        console.print(Panel(
            "\n".join(f"• {n}" for n in names),
            title="Available tools",
            border_style="dim",
        ))
        return True

    elif cmd_lower == "/think":
        think_next = True
        console.print("[dim]Thinking on for next message (play mode only).[/dim]")
        return True

    elif cmd_lower == "/help":
        console.print(Panel(
            "[bold]/quit[/bold]    — Exit\n"
            "[bold]/clear[/bold]   — Reset conversation\n"
            "[bold]/model[/bold]   — Show model + mode info\n"
            "[bold]/work[/bold]    — Switch to tool-using mode\n"
            "[bold]/play[/bold]    — Switch to persona mode (default)\n"
            "[bold]/tools[/bold]   — List available tools\n"
            "[bold]/think[/bold]   — Show thinking on next message (play mode)\n"
            "[bold]/help[/bold]    — This message",
            title="Commands",
            border_style="dim",
        ))
        return True

    return False


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    console.print(Panel(
        f"[bold]Tort Agent[/bold] — {MODEL}\n"
        f"[dim]/work for tools • /help for commands[/dim]",
        border_style="blue",
    ))

    while True:
        try:
            prompt_color = "magenta" if mode == "work" else "blue"
            user_input = console.input(
                f"\n[bold {prompt_color}]>>> [/bold {prompt_color}]"
            ).strip()

            if not user_input:
                continue

            if user_input.startswith("/"):
                if handle_command(user_input):
                    continue

            chat(user_input)

        except KeyboardInterrupt:
            console.print("\n[dim]Interrupted. Type /quit to exit.[/dim]")
            continue


if __name__ == "__main__":
    main()
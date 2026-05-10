# Tort Agent

Local LLM agent with a two-mode interface — persona chat and tool-using study assistant — running on `qwen3.6:35b-a3b` via Ollama. No cloud LLMs, no SaaS dependencies. All inference runs on owned hardware.

---

## Architecture

```
    ┌────────────────────────────────────────────────────────────────────┐
    │                         User Interface                             │
    │                                                                    │
    │   Terminal REPL ──► agent.py ──► mode router                       │
    │                                    │                               │
    │                        ┌───────────┴───────────┐                   │
    │                        │                       │                   │
    │                   /play mode              /work mode               │
    │                   (persona)              (tool-use)                │
    │                        │                       │                   │
    │                   Streamed tokens         Batch response           │
    │                   Socratic persona        Lean study prompt        │
    │                   No tool calls           Tool-use loop            │
    └────────────────────┬───────────────────────────┬───────────────────┘
                         │                           │
    ┌────────────────────┴───────────────────────────┴───────────────────┐
    │                      Inference (Ollama)                             │
    │                                                                    │
    │   qwen3.6:35b-a3b (tower, RTX 4090)                               │
    │   qwen3.5:9b (MacBook, local dev)                                  │
    │                                                                    │
    │   Native tool_call support ──► TOOL_REGISTRY dispatch              │
    └────────────────────────────────────┬──────────────────────────────┘
                                         │
    ┌────────────────────────────────────┴──────────────────────────────┐
    │                      Tool Layer (tools.py)                        │
    │                                                                    │
    │   morning_brief ──► wttr.in + Canvas API + vault scan             │
    │   export_apple_notes ──► AppleScript → iCloud staging             │
    │   convert_notes ──► PyMuPDF → Apple Vision OCR → Ollama cleanup  │
    │   develop_concepts ──► JSON extraction → 01-concepts/ with dedup │
    │   weekly_summary ──► tag-filtered scan → 05-logs/                 │
    │   list_directory ──► vault navigation primitive                    │
    │                                                                    │
    │   All tools: read env → call API/fs → format via Ollama → return  │
    └────────────────────────────────────┬──────────────────────────────┘
                                         │
    ┌────────────────────────────────────┴──────────────────────────────┐
    │                    Data Layer                                      │
    │                                                                    │
    │   Obsidian Vault (/Users/tortellini/tort-vault)                   │
    │   ├── 01-concepts/    atomic concept files with frontmatter       │
    │   ├── 03-courses/     per-course lecture notes                    │
    │   ├── 05-logs/        weekly summaries                            │
    │   ├── CLAUDE.md       vault conventions                           │
    │   └── _lessons.md     project-level lessons learned               │
    │                                                                    │
    │   Canvas LMS API ──► assignment deadlines + grades                │
    │   Apple Notes (iCloud) ──► handwritten note PDFs                  │
    └───────────────────────────────────────────────────────────────────┘
```

---

## Mode Design

| Mode | System Prompt | Streaming | Tool Calls | Use Case |
|------|---------------|-----------|------------|----------|
| `/play` | The Arbiter Mentis (noir / Socratic / Stoic persona) | Yes | No | Conversation, philosophy, tutoring |
| `/work` | Lean study-assistant prompt | No (batch) | Yes | Any task requiring a tool |

The two-mode split is a practical design decision: streaming + tool-calling don't compose cleanly when `<think>` blocks also need to be stripped for display. Separating modes keeps each path robust and lets `/play` push the persona hard without compromising tool reliability in `/work`.

---

## Tool-Use Loop (`/work` mode)

The agentic loop in `agent.py` handles the bookkeeping that separates a working agent from a demo:

- **Hard iteration cap** (`MAX_TOOL_ITERATIONS = 5`) — prevents runaway tool-call loops
- **JSON-string argument coercion** — Ollama sometimes returns tool args as a JSON string, sometimes as a dict; both shapes are handled
- **Working-copy message list** — tool exchanges don't pollute long-term conversation history
- **`<think>` block stripping** — streaming buffer in `/play` mode handles `<` characters in code/HTML without false-triggering
- **Connection/timeout recovery** — pops the user message off history on a failed turn rather than leaving it dangling

---

## Setup

```bash
git clone https://github.com/Cap-Dylan/tort-agent.git && cd tort-agent
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Set: CANVAS_API_TOKEN, CANVAS_BASE_URL, OBSIDIAN_VAULT,
#      OLLAMA_URL, OLLAMA_MODEL

python3 agent.py
```

**REPL commands**: `/work`, `/play`, `/tools`, `/think`, `/clear`, `/model`, `/help`, `/quit`

---

## Phase Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | All six tool implementations | ✅ Complete |
| 2 | Tool-use loop, mode toggle, structured-output parsing, error handling | ✅ Complete |
| 3 | Eval harness for tool-call accuracy across model variants | ✅ Complete — 3 runs on MSI (3-trial + 10-trial), results committed |
| 4 | macOS Shortcut surface — invoke agent from anywhere on the system | Planned |

---

## Hardware

Designed for tower deployment, developed on the MacBook:

| Node | Model | Role | Performance |
|------|-------|------|-------------|
| **Tower** (production) | `qwen3.6:35b-a3b` | Full tool-use agent | RTX 4090 24GB, 128GB DDR5 — full model in VRAM |
| **MacBook** (dev) | `qwen3.5:9b` | Lightweight local testing | M4 Pro, 24GB unified memory |
| **MSI** (eval) | `qwen3.5:9b` | Eval harness runs | RTX 2060 8GB — 22 tok/s, 1.0 keyword recall, 0.85 OCR quality |

The agent is host-agnostic — set `OLLAMA_URL` and `OLLAMA_MODEL` in `.env` and it works against any Ollama instance with a tool-calling-capable model.

---

## License

MIT — see [LICENSE](LICENSE).

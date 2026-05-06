# llm-testing-framework

A terminal-based tool for comparing OpenAI and Google Gemini responses side by side in real time. Send a message once and see both models reply in split panels — useful for prompt engineering, model evaluation, and behavior comparison.

---

## Demo

> _Screenshot placeholder — run `llm-tester chat` to see the split-panel UI._

---

## Features

- Split-panel TUI showing OpenAI (left) and Gemini (right) responses simultaneously
- Concurrent API calls via `asyncio.gather` — no waiting for one model before the other
- Separate conversation histories maintained per model
- Optional system prompt loaded from a file
- Conversation log written to `chat_log.txt` in the working directory
- Reset conversation mid-session without restarting the process

---

## Prerequisites

- Python 3.11+
- An [OpenAI API key](https://platform.openai.com/api-keys)
- A [Google Gemini API key](https://aistudio.google.com/app/apikey)

---

## Installation

```bash
git clone https://github.com/talhanadeem/llm-testing-framework.git
cd llm-testing-framework

cp .env.example .env
# Edit .env and add your API keys

pip install -e .
```

---

## Usage

```bash
llm-tester chat
```

### Options

| Option | Default | Description |
|---|---|---|
| `--openai-model` | `gpt-5.4-mini` | OpenAI model to use |
| `--gemini-model` | `gemini-3-flash-preview` | Gemini model to use |
| `--system-prompt-file` | `system_prompt.txt` | Path to a system prompt file |

### Examples

```bash
# Use specific models
llm-tester chat --openai-model gpt-4o --gemini-model gemini-1.5-pro

# Load a custom system prompt
llm-tester chat --system-prompt-file path/to/prompt.txt
```

---

## Keyboard Shortcuts

| Shortcut | Action |
|---|---|
| `Ctrl+C` | Quit |
| `Ctrl+R` | Reset conversation (clears both histories and panels) |

---

## Configuration

### Environment Variables (`.env`)

```
OPENAI_API_KEY=your_openai_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```

Copy `.env.example` to `.env` and fill in your keys. The file is loaded automatically on startup.

### System Prompt

Place a system prompt in `system_prompt.txt` (or any path passed to `--system-prompt-file`). If the file is not found, the tool starts without a system prompt.

---

## Architecture

The project follows a three-layer design:

1. **CLI** (`llm_tester/cli.py`) — Click-based entry point. Handles argument parsing, `.env` loading, and system prompt file reading.

2. **TUI** (`llm_tester/tui.py`) — Textual-based split-panel terminal app (`ChatApp`). Left panel shows OpenAI responses, right panel shows Gemini. Maintains separate message history lists per model. Calls both APIs concurrently via `asyncio.gather()`.

3. **Providers** (`llm_tester/providers.py`) — Thin wrappers around the OpenAI and Google Gemini clients. Both `call_openai()` and `call_gemini()` share the same `(messages, model, system_prompt)` signature and are safe to call via `asyncio.to_thread()`.

**Data flow:** user types in Input → `on_input_submitted` appends to both history lists → `asyncio.gather(call_openai, call_gemini)` → responses written to respective `RichLog` panels → exchange appended to `chat_log.txt`.

---

## Dependencies

| Package | Version |
|---|---|
| `openai` | >=1.30.0 |
| `google-genai` | >=1.0.0 |
| `rich` | >=13.0.0 |
| `click` | >=8.1.0 |
| `python-dotenv` | >=1.0.0 |
| `textual` | >=0.70.0 |

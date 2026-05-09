# llm-testing-framework

A terminal-based tool for comparing LLM responses side by side and running automated batch tests against multiple models simultaneously. Supports OpenAI, Google Gemini, Qwen (DashScope), and DeepSeek.

---

## Features

- Split-panel TUI showing OpenAI (left) and Gemini (right) responses simultaneously
- Concurrent API calls via `asyncio.gather` — no waiting for one model before the other
- Separate conversation histories maintained per model
- Optional system prompt loaded from a file
- Conversation log written to `chat_log.txt` in the working directory
- Reset conversation mid-session without restarting the process
- **Batch intent classification** — run test cases against all four models and compare results
- **Batch execution testing** — evaluate tool-calling behavior across models
- **Batch composer testing** — LLM-as-judge evaluation of response quality

---

## Prerequisites

- Python 3.11+
- API keys for the providers you want to use (see [Configuration](#configuration))

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

## Commands

### `chat` — Interactive split-panel chat

Compare OpenAI and Gemini side by side in real time.

```bash
llm-tester chat
llm-tester chat --openai-model gpt-4o --gemini-model gemini-1.5-pro
llm-tester chat --system-prompt-file path/to/prompt.txt
```

| Option | Default | Description |
|---|---|---|
| `--openai-model` | `gpt-5.4-mini` | OpenAI model to use |
| `--gemini-model` | `gemini-3-flash-preview` | Gemini model to use |
| `--system-prompt-file` | `system_prompt.txt` | Path to a system prompt file |

**Keyboard shortcuts:**

| Shortcut | Action |
|---|---|
| `Ctrl+C` | Quit |
| `Ctrl+R` | Reset conversation (clears both histories and panels) |

---

### `tonality` — Tonality Agent interactive chat

Interactive chat for testing the Tonality Agent. Loads `Tonality Agent/system_prompt.txt` by default.

```bash
llm-tester tonality
llm-tester tonality --system-prompt-file "Tonality Agent/system_prompt.txt"
```

| Option | Default | Description |
|---|---|---|
| `--openai-model` | `gpt-5.4-mini` | OpenAI model to use |
| `--gemini-model` | `gemini-3-flash-preview` | Gemini model to use |
| `--system-prompt-file` | `Tonality Agent/system_prompt.txt` | Path to the system prompt file |

---

### `batch` — Intent classification batch tests

Run a JSON test suite against OpenAI, Gemini, Qwen, and DeepSeek. Compares predicted intents (and optionally confidence levels) against expected values.

```bash
llm-tester batch
llm-tester batch --check-confidence --output-file results.txt
llm-tester batch "Test Parsing Agent/test_cases.json"
```

| Option | Default | Description |
|---|---|---|
| `--openai-model` | `gpt-5.4-mini` | OpenAI model |
| `--gemini-model` | `gemini-3-flash-preview` | Gemini model |
| `--qwen-model` | `qwen3.5-flash` | Qwen model |
| `--deepseek-model` | `deepseek-v4-flash` | DeepSeek model |
| `--test-file` | `Test Parsing Agent/test_cases.json` | Path to JSON test cases |
| `--system-prompt-file` | `Test Parsing Agent/system_prompt.txt` | Path to the system prompt |
| `--check-confidence` | off | Also validate confidence level, not just intent |
| `--output-file` | — | Save results to a `.txt` file |

---

### `batch-execution` — Tool-calling execution batch tests

Evaluate tool-calling (function-calling) behavior of all four models against the Test Execution Agent's test suite.

```bash
llm-tester batch-execution
llm-tester batch-execution --output-file execution_results.txt
```

| Option | Default | Description |
|---|---|---|
| `--openai-model` | `gpt-5.4-mini` | OpenAI model |
| `--gemini-model` | `gemini-3-flash-preview` | Gemini model |
| `--qwen-model` | `qwen3.5-flash` | Qwen model |
| `--deepseek-model` | `deepseek-v4-flash` | DeepSeek model |
| `--test-file` | `Test Execution agent/test_cases.json` | Path to JSON test cases |
| `--system-prompt-file` | `Test Execution agent/system_prompt.txt` | Path to the system prompt |
| `--output-file` | — | Save results to a `.txt` file |

---

### `batch-composer` — LLM-as-judge composer batch tests

Evaluate Composer Agent response quality using an OpenAI model as a judge. Each candidate model's output is scored by the judge against expected criteria from the test cases.

```bash
llm-tester batch-composer
llm-tester batch-composer --judge-model gpt-4o --output-file composer_results.txt
llm-tester batch-composer "Composer Agent/test_cases.json"
```

| Option | Default | Description |
|---|---|---|
| `--openai-model` | `gpt-5.4-mini` | OpenAI model as composer |
| `--gemini-model` | `gemini-3-flash-preview` | Gemini model as composer |
| `--qwen-model` | `qwen3.5-flash` | Qwen model as composer |
| `--deepseek-model` | `deepseek-v4-flash` | DeepSeek model as composer |
| `--judge-model` | `gpt-5.4-mini` | OpenAI model to use as judge |
| `--system-prompt-file` | `Composer Agent/system_prompt.txt` | Path to the composer system prompt |
| `--output-file` | — | Save results to a `.txt` file |

---

## Configuration

### Environment Variables (`.env`)

```
OPENAI_API_KEY=your_openai_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
DASHSCOPE_API_KEY=your_dashscope_api_key_here
DEEPSEEK_API_KEY=your_deepseek_api_key_here
```

Copy `.env.example` to `.env` and fill in the keys for the providers you intend to use. The file is loaded automatically on startup.

- **OpenAI** — [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
- **Gemini** — [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
- **Qwen (DashScope)** — [dashscope.aliyuncs.com](https://dashscope.aliyuncs.com)
- **DeepSeek** — [platform.deepseek.com](https://platform.deepseek.com)

### Agent directories

Each agent has its own directory containing a `system_prompt.txt` and `test_cases.json`. These are ignored by git — you manage them locally.

| Directory | Used by |
|---|---|
| `Tonality Agent/` | `tonality` command |
| `Test Parsing Agent/` | `batch` command |
| `Test Execution agent/` | `batch-execution` command |
| `Composer Agent/` | `batch-composer` command |

---

## Architecture

Three-layer design:

1. **CLI** (`llm_tester/cli.py`) — Click-based entry point. Handles argument parsing, `.env` loading, system prompt file reading, and test case loading. Dispatches to either the TUI or a batch runner.

2. **TUI** (`llm_tester/tui.py`) — Textual-based split-panel terminal app (`ChatApp`). Left panel shows OpenAI responses, right panel shows Gemini. Maintains separate message history lists per model. Calls both APIs concurrently via `asyncio.gather()`.

3. **Providers** (`llm_tester/providers.py`) — Thin wrappers around four model APIs. All functions (`call_openai`, `call_gemini`, `call_qwen`, `call_deepseek`) share the same `(messages, model, system_prompt)` signature and are safe to call via `asyncio.to_thread()`.

4. **Batch runners** — Three modules handle different test types:
   - `llm_tester/batch.py` — intent classification, JSON response parsing, pass/fail per model
   - `llm_tester/execution_batch.py` — tool-calling evaluation
   - `llm_tester/composer_batch.py` — LLM-as-judge scoring

**Data flow (chat):** user types → `on_input_submitted` appends to both histories → `asyncio.gather(call_openai, call_gemini)` → responses written to `RichLog` panels → exchange appended to `chat_log.txt`.

**Data flow (batch):** test cases loaded from JSON → all models called concurrently per test case → results compared against expected values → summary table printed (and optionally saved to file).

---

## Dependencies

| Package | Version |
|---|---|
| `openai` | >=1.30.0 |
| `google-genai` | >=1.0.0 |
| `anthropic` | >=0.30.0 |
| `rich` | >=13.0.0 |
| `click` | >=8.1.0 |
| `python-dotenv` | >=1.0.0 |
| `textual` | >=0.70.0 |

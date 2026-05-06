import asyncio

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Input, RichLog, Static

from llm_tester.providers import call_openai, call_gemini


class ChatApp(App):
    CSS = """
    Screen {
        layout: vertical;
    }

    #panels {
        height: 1fr;
    }

    .model-panel {
        width: 1fr;
        border: solid $primary;
        padding: 0 1;
    }

    .panel-title {
        text-align: center;
        background: $primary;
        color: $text;
        padding: 0 1;
        margin-bottom: 1;
    }

    RichLog {
        height: 1fr;
    }

    #input-bar {
        height: auto;
        border-top: solid $primary;
        padding: 0 1;
    }
    """

    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+r", "reset", "Reset conversation"),
    ]

    def __init__(self, openai_model: str, gemini_model: str, system_prompt: str):
        super().__init__()
        self.openai_model = openai_model
        self.gemini_model = gemini_model
        self.system_prompt = system_prompt
        self.openai_history: list[dict] = []
        self.gemini_history: list[dict] = []
        self._log_path = "chat_log.txt"
        self._conversation_count = 0

    def on_mount(self) -> None:
        self._start_new_conversation()

    def _start_new_conversation(self) -> None:
        self._conversation_count += 1
        from datetime import datetime
        header = (
            f"\n=====================================\n"
            f"Conversation {self._conversation_count} — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"=====================================\n"
        )
        try:
            with open(self._log_path, "a") as f:
                f.write(header)
        except OSError:
            pass

    def _log_exchange(self, user_msg: str, openai_resp: str, gemini_resp: str) -> None:
        entry = (
            f"\n[You]: {user_msg}\n\n"
            f"[OpenAI - {self.openai_model}]: {openai_resp}\n\n"
            f"[Gemini - {self.gemini_model}]: {gemini_resp}\n"
        )
        try:
            with open(self._log_path, "a") as f:
                f.write(entry)
        except OSError:
            pass

    def compose(self) -> ComposeResult:
        with Horizontal(id="panels"):
            with Vertical(classes="model-panel"):
                yield Static(f"OpenAI ({self.openai_model})", classes="panel-title")
                yield RichLog(id="openai-log", wrap=True, highlight=True, markup=True)
            with Vertical(classes="model-panel"):
                yield Static(f"Gemini ({self.gemini_model})", classes="panel-title")
                yield RichLog(id="gemini-log", wrap=True, highlight=True, markup=True)
        with Vertical(id="input-bar"):
            yield Input(placeholder="Type your message and press Enter...")

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        message = event.value.strip()
        if not message:
            return

        user_input = self.query_one(Input)
        openai_log = self.query_one("#openai-log", RichLog)
        gemini_log = self.query_one("#gemini-log", RichLog)

        user_input.clear()
        user_input.disabled = True
        user_input.placeholder = "Thinking..."

        openai_log.write(f"[bold cyan]You:[/bold cyan] {message}")
        gemini_log.write(f"[bold cyan]You:[/bold cyan] {message}")

        self.openai_history.append({"role": "user", "content": message})
        self.gemini_history.append({"role": "user", "content": message})

        openai_result, gemini_result = await asyncio.gather(
            asyncio.to_thread(call_openai, list(self.openai_history), self.openai_model, self.system_prompt),
            asyncio.to_thread(call_gemini, list(self.gemini_history), self.gemini_model, self.system_prompt),
            return_exceptions=True,
        )

        if isinstance(openai_result, Exception):
            openai_response = f"[red]Error: {openai_result}[/red]"
        else:
            openai_response = openai_result
            self.openai_history.append({"role": "assistant", "content": openai_response})

        if isinstance(gemini_result, Exception):
            gemini_response = f"[red]Error: {gemini_result}[/red]"
        else:
            gemini_response = gemini_result
            self.gemini_history.append({"role": "assistant", "content": gemini_response})

        openai_log.write(f"[bold green]AI:[/bold green] {openai_response}")
        openai_log.write("")
        gemini_log.write(f"[bold blue]AI:[/bold blue] {gemini_response}")
        gemini_log.write("")

        openai_log_text = f"Error: {openai_result}" if isinstance(openai_result, Exception) else openai_result
        gemini_log_text = f"Error: {gemini_result}" if isinstance(gemini_result, Exception) else gemini_result
        self._log_exchange(message, openai_log_text, gemini_log_text)

        user_input.disabled = False
        user_input.placeholder = "Type your message and press Enter..."
        user_input.focus()

    def action_reset(self) -> None:
        self.openai_history.clear()
        self.gemini_history.clear()
        self.query_one("#openai-log", RichLog).clear()
        self.query_one("#gemini-log", RichLog).clear()
        self._start_new_conversation()

import click
from dotenv import load_dotenv

from llm_tester.tui import ChatApp

load_dotenv()


@click.group()
def main():
    pass


@main.command()
@click.option("--openai-model", default="gpt-5.4-mini", show_default=True, help="OpenAI model to use.")
@click.option("--gemini-model", default="gemini-3-flash-preview", show_default=True, help="Gemini model to use.")
@click.option("--system-prompt-file", default="system_prompt.txt", show_default=True, help="Path to the system prompt file.")
def chat(openai_model, gemini_model, system_prompt_file):
    """Interactive split-panel chat comparing OpenAI and Gemini side by side."""
    system_prompt = ""
    try:
        with open(system_prompt_file) as f:
            system_prompt = f.read().strip()
    except FileNotFoundError:
        pass

    app = ChatApp(openai_model=openai_model, gemini_model=gemini_model, system_prompt=system_prompt)
    app.run()

import asyncio
import json

import click
from dotenv import load_dotenv

from llm_tester.tui import ChatApp

load_dotenv()


def _load_system_prompt(path: str) -> str:
    try:
        with open(path) as f:
            return f.read().strip()
    except FileNotFoundError:
        return ""


@click.group()
def main():
    pass


@main.command()
@click.option("--openai-model", default="gpt-5.4-mini", show_default=True, help="OpenAI model to use.")
@click.option("--gemini-model", default="gemini-3-flash-preview", show_default=True, help="Gemini model to use.")
@click.option("--system-prompt-file", default="system_prompt.txt", show_default=True, help="Path to the system prompt file.")
def chat(openai_model, gemini_model, system_prompt_file):
    """Interactive split-panel chat comparing OpenAI and Gemini side by side."""
    system_prompt = _load_system_prompt(system_prompt_file)
    app = ChatApp(openai_model=openai_model, gemini_model=gemini_model, system_prompt=system_prompt)
    app.run()


@main.command()
@click.option("--openai-model", default="gpt-5.4-mini", show_default=True, help="OpenAI model to use.")
@click.option("--gemini-model", default="gemini-3-flash-preview", show_default=True, help="Gemini model to use.")
@click.option("--system-prompt-file", default="Tonality Agent/system_prompt.txt", show_default=True, help="Path to the system prompt file.")
def tonality(openai_model, gemini_model, system_prompt_file):
    """Interactive chat for testing the Tonality Agent."""
    system_prompt = _load_system_prompt(system_prompt_file)
    app = ChatApp(openai_model=openai_model, gemini_model=gemini_model, system_prompt=system_prompt)
    app.run()


@main.command()
@click.option("--openai-model", default="gpt-5.4-mini", show_default=True, help="OpenAI model to use.")
@click.option("--gemini-model", default="gemini-3-flash-preview", show_default=True, help="Gemini model to use.")
@click.option("--qwen-model", default="qwen3.5-flash", show_default=True, help="Qwen model to use.")
@click.option("--deepseek-model", default="deepseek-v4-flash", show_default=True, help="DeepSeek model to use.")
@click.option("--anthropic-model", default="claude-sonnet-4-6", show_default=True, help="Anthropic model to use.")
@click.option("--test-file", default="Test Parsing Agent/test_cases.json", show_default=True, help="Path to JSON test cases file.")
@click.option("--system-prompt-file", default="Test Parsing Agent/system_prompt.txt", show_default=True, help="Path to the system prompt file.")
@click.option("--check-confidence", is_flag=True, default=False, help="Also check confidence level, not just intent.")
@click.option("--output-file", default=None, help="Path to save results as a .txt file.")
def batch(openai_model, gemini_model, qwen_model, deepseek_model, anthropic_model, test_file, system_prompt_file, check_confidence, output_file):
    """Run batch intent classification tests against OpenAI, Gemini, Qwen, DeepSeek, and Anthropic."""
    from llm_tester.batch import run_batch

    system_prompt = _load_system_prompt(system_prompt_file)

    try:
        with open(test_file) as f:
            test_cases = json.load(f)
    except FileNotFoundError:
        raise click.ClickException(f"Test file not found: {test_file}")
    except json.JSONDecodeError as e:
        raise click.ClickException(f"Invalid JSON in test file: {e}")

    asyncio.run(run_batch(test_cases, openai_model, gemini_model, qwen_model, system_prompt, check_confidence, output_file, deepseek_model, anthropic_model))


@main.command("batch-execution")
@click.option("--openai-model", default="gpt-5.4-mini", show_default=True, help="OpenAI model to use.")
@click.option("--gemini-model", default="gemini-3-flash-preview", show_default=True, help="Gemini model to use.")
@click.option("--qwen-model", default="qwen3.5-flash", show_default=True, help="Qwen model to use.")
@click.option("--deepseek-model", default="deepseek-v4-flash", show_default=True, help="DeepSeek model to use.")
@click.option("--anthropic-model", default="claude-sonnet-4-6", show_default=True, help="Anthropic model to use.")
@click.option("--test-file", default="Test Execution agent/test_cases.json", show_default=True, help="Path to JSON test cases file.")
@click.option("--system-prompt-file", default="Test Execution agent/system_prompt.txt", show_default=True, help="Path to the system prompt file.")
@click.option("--output-file", default=None, help="Path to save results as a .txt file.")
def batch_execution(openai_model, gemini_model, qwen_model, deepseek_model, anthropic_model, test_file, system_prompt_file, output_file):
    """Run batch tool-calling tests for the Execution Agent against OpenAI, Gemini, Qwen, DeepSeek, and Anthropic."""
    from llm_tester.execution_batch import run_execution_batch

    system_prompt = _load_system_prompt(system_prompt_file)

    try:
        with open(test_file) as f:
            test_cases = json.load(f)
    except FileNotFoundError:
        raise click.ClickException(f"Test file not found: {test_file}")
    except json.JSONDecodeError as e:
        raise click.ClickException(f"Invalid JSON in test file: {e}")

    asyncio.run(run_execution_batch(test_cases, openai_model, gemini_model, system_prompt, output_file, qwen_model, deepseek_model, anthropic_model))


@main.command("batch-composer")
@click.argument("test_file", default="Composer Agent/test_cases.json")
@click.option("--system-prompt-file", default="Composer Agent/system_prompt.txt", show_default=True, help="Path to the composer system prompt file.")
@click.option("--openai-model", default="gpt-5.4-mini", show_default=True, help="OpenAI model to use as composer.")
@click.option("--gemini-model", default="gemini-3-flash-preview", show_default=True, help="Gemini model to use as composer.")
@click.option("--qwen-model", default="qwen3.5-flash", show_default=True, help="Qwen model to use as composer.")
@click.option("--deepseek-model", default="deepseek-v4-flash", show_default=True, help="DeepSeek model to use as composer.")
@click.option("--anthropic-model", default="claude-sonnet-4-6", show_default=True, help="Anthropic model to use as composer.")
@click.option("--judge-model", default="gpt-5.4-mini", show_default=True, help="OpenAI model to use as judge.")
@click.option("--output-file", default=None, help="Path to save results as a .txt file.")
def batch_composer(test_file, system_prompt_file, openai_model, gemini_model, qwen_model, deepseek_model, anthropic_model, judge_model, output_file):
    """Run LLM-as-judge batch tests for the Composer Agent against OpenAI, Gemini, Qwen, DeepSeek, and Anthropic."""
    from llm_tester.composer_batch import run_composer_batch

    system_prompt = _load_system_prompt(system_prompt_file)

    try:
        with open(test_file) as f:
            test_cases = json.load(f)
    except FileNotFoundError:
        raise click.ClickException(f"Test file not found: {test_file}")
    except json.JSONDecodeError as e:
        raise click.ClickException(f"Invalid JSON in test file: {e}")

    asyncio.run(run_composer_batch(test_cases, openai_model, gemini_model, system_prompt, judge_model, output_file, qwen_model, deepseek_model, anthropic_model))

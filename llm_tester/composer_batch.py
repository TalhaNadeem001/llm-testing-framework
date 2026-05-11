import asyncio
import json
import re
from datetime import datetime, timezone

from rich.console import Console
from rich.table import Table
from rich import box

from llm_tester.providers import call_openai, call_gemini, call_qwen, call_deepseek, call_anthropic


JUDGE_SYSTEM_PROMPT = (
    "You are an expert evaluator assessing whether a customer service agent response "
    "correctly handles a given scenario. Be strict but fair. "
    "Return ONLY valid JSON with no extra text."
)


def _format_conversation(conversation: list[dict]) -> str:
    lines = []
    for msg in conversation:
        role = msg["role"].capitalize()
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def _call_composer(conversation: list[dict], model: str, system_prompt: str) -> str:
    """Send the full conversation to the composer model and return its response."""
    return call_openai(conversation, model, system_prompt) if "gpt" in model.lower() \
        else call_gemini(conversation, model, system_prompt)


def _call_composer_openai(conversation: list[dict], model: str, system_prompt: str) -> str:
    return call_openai(conversation, model, system_prompt)


def _call_composer_gemini(conversation: list[dict], model: str, system_prompt: str) -> str:
    return call_gemini(conversation, model, system_prompt)


def _call_composer_qwen(conversation: list[dict], model: str, system_prompt: str) -> str:
    return call_qwen(conversation, model, system_prompt)


def _call_composer_deepseek(conversation: list[dict], model: str, system_prompt: str) -> str:
    return call_deepseek(conversation, model, system_prompt)


def _call_composer_anthropic(conversation: list[dict], model: str, system_prompt: str) -> str:
    return call_anthropic(conversation, model, system_prompt)


def _judge_response(
    scenario: str,
    conversation: list[dict],
    response: str,
    expected_behaviors: list[str],
    judge_model: str,
) -> dict:
    """Use OpenAI as judge to evaluate the composer's response against expected behaviors."""
    criteria_list = "\n".join(f"- {b}" for b in expected_behaviors)
    formatted_conv = _format_conversation(conversation)

    judge_user_prompt = (
        f"You are evaluating whether an AI agent's response correctly follows its scenario.\n\n"
        f"Scenario: {scenario}\n\n"
        f"Conversation:\n{formatted_conv}\n\n"
        f"Agent response: {response}\n\n"
        f"Check each criterion below and return JSON in this exact format:\n"
        f'{{"criteria": [{{"criterion": "...", "met": true, "explanation": "..."}}], "overall_pass": true}}\n\n'
        f"Criteria to check:\n{criteria_list}"
    )

    raw = call_openai(
        [{"role": "user", "content": judge_user_prompt}],
        judge_model,
        JUDGE_SYSTEM_PROMPT,
    )

    # Strip markdown fences
    clean = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {"criteria": [], "overall_pass": False, "parse_error": True}


def _failed_criteria(judgment: dict) -> list[str]:
    return [
        c["criterion"]
        for c in judgment.get("criteria", [])
        if not c.get("met", False)
    ]


def _fmt_failed(criteria: list[str], max_len: int = 50) -> str:
    if not criteria:
        return ""
    joined = "; ".join(criteria)
    return joined if len(joined) <= max_len else joined[:max_len - 3] + "..."


def _write_txt(
    output_file: str,
    cases: list[dict],
    total: int,
    oai_pass: int,
    gem_pass: int,
    qwn_pass: int,
    dsk_pass: int,
    ant_pass: int,
) -> None:
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines = [
        f"Composer Batch Results — {timestamp}",
        "=" * 60,
        "",
    ]
    for c in cases:
        lines.append("=" * 64)
        lines.append(f"[{c['id']}]  {c['flow']}")
        lines.append("=" * 64)
        lines.append("")

        # Conversation
        lines.append("CONVERSATION:")
        for msg in c.get("conversation", []):
            role = msg["role"].capitalize()
            lines.append(f"  {role}: {msg['content']}")
        lines.append("")

        # Expected behaviors
        lines.append("EXPECTED BEHAVIORS:")
        for behavior in c.get("expected_behaviors", []):
            lines.append(f"  • {behavior}")
        lines.append("")

        # Per-model results
        models = [
            ("OpenAI",    c["oai_ok"], c.get("oai_response", ""), c.get("oai_criteria", [])),
            ("Gemini",    c["gem_ok"], c.get("gem_response", ""), c.get("gem_criteria", [])),
            ("Qwen",      c["qwn_ok"], c.get("qwn_response", ""), c.get("qwn_criteria", [])),
            ("DeepSeek",  c["dsk_ok"], c.get("dsk_response", ""), c.get("dsk_criteria", [])),
            ("Anthropic", c["ant_ok"], c.get("ant_response", ""), c.get("ant_criteria", [])),
        ]
        for name, ok, response, criteria in models:
            tick = "✓" if ok else "✗"
            lines.append(f"--- {name}: {tick} ---")
            lines.append(f'Response: "{response}"')
            failed = [(cr["criterion"], cr.get("explanation", "")) for cr in criteria if not cr.get("met", True)]
            if failed:
                lines.append("Failed:")
                for criterion, explanation in failed:
                    suffix = f" — {explanation}" if explanation else ""
                    lines.append(f"  • {criterion}{suffix}")
            lines.append("")

        lines.append("")

    lines += [
        "-" * 60,
        f"OpenAI: {oai_pass}/{total} ({oai_pass/total*100:.1f}%)  |  Gemini: {gem_pass}/{total} ({gem_pass/total*100:.1f}%)  |  Qwen: {qwn_pass}/{total} ({qwn_pass/total*100:.1f}%)  |  DeepSeek: {dsk_pass}/{total} ({dsk_pass/total*100:.1f}%)  |  Anthropic: {ant_pass}/{total} ({ant_pass/total*100:.1f}%)",
    ]
    with open(output_file, "w") as f:
        f.write("\n".join(lines) + "\n")


async def run_composer_batch(
    test_cases: list[dict],
    openai_model: str,
    gemini_model: str,
    system_prompt: str,
    judge_model: str,
    output_file: str | None = None,
    qwen_model: str = "qwen3.5-flash",
    deepseek_model: str = "deepseek-v4-flash",
    anthropic_model: str = "claude-haiku-4-5-20251001",
) -> None:
    console = Console()

    table = Table(box=box.SIMPLE_HEAVY, show_lines=True)
    table.add_column("ID", style="bold cyan", no_wrap=True, width=5)
    table.add_column("Flow / Scenario", max_width=36)
    table.add_column("OAI", justify="center", no_wrap=True, width=5)
    table.add_column("GEM", justify="center", no_wrap=True, width=5)
    table.add_column("QWN", justify="center", no_wrap=True, width=5)
    table.add_column("DSK", justify="center", no_wrap=True, width=5)
    table.add_column("ANT", justify="center", no_wrap=True, width=5)
    table.add_column("Failed Criteria", max_width=50)

    oai_pass = 0
    gem_pass = 0
    qwn_pass = 0
    dsk_pass = 0
    ant_pass = 0
    plain_rows = []
    total = len(test_cases)

    console.print(f"\n[bold]Running {total} composer test cases...[/bold]\n")

    for i, case in enumerate(test_cases, 1):
        case_id = case.get("id", "?")
        flow = case.get("flow", "")
        scenario = case.get("scenario", "")
        conversation = case.get("conversation", [])
        expected_behaviors = case.get("expected_behaviors", [])

        console.print(f"  [{i}/{total}] {case_id}: {scenario}...", end=" ")

        # Call all five models concurrently
        oai_response, gem_response, qwn_response, dsk_response, ant_response = await asyncio.gather(
            asyncio.to_thread(_call_composer_openai, conversation, openai_model, system_prompt),
            asyncio.to_thread(_call_composer_gemini, conversation, gemini_model, system_prompt),
            asyncio.to_thread(_call_composer_qwen, conversation, qwen_model, system_prompt),
            asyncio.to_thread(_call_composer_deepseek, conversation, deepseek_model, system_prompt),
            asyncio.to_thread(_call_composer_anthropic, conversation, anthropic_model, system_prompt),
        )

        # Judge all five responses concurrently
        oai_judgment, gem_judgment, qwn_judgment, dsk_judgment, ant_judgment = await asyncio.gather(
            asyncio.to_thread(_judge_response, scenario, conversation, oai_response, expected_behaviors, judge_model),
            asyncio.to_thread(_judge_response, scenario, conversation, gem_response, expected_behaviors, judge_model),
            asyncio.to_thread(_judge_response, scenario, conversation, qwn_response, expected_behaviors, judge_model),
            asyncio.to_thread(_judge_response, scenario, conversation, dsk_response, expected_behaviors, judge_model),
            asyncio.to_thread(_judge_response, scenario, conversation, ant_response, expected_behaviors, judge_model),
        )

        oai_ok = oai_judgment.get("overall_pass", False)
        gem_ok = gem_judgment.get("overall_pass", False)
        qwn_ok = qwn_judgment.get("overall_pass", False)
        dsk_ok = dsk_judgment.get("overall_pass", False)
        ant_ok = ant_judgment.get("overall_pass", False)

        if oai_ok:
            oai_pass += 1
        if gem_ok:
            gem_pass += 1
        if qwn_ok:
            qwn_pass += 1
        if dsk_ok:
            dsk_pass += 1
        if ant_ok:
            ant_pass += 1

        oai_failed = _failed_criteria(oai_judgment)
        gem_failed = _failed_criteria(gem_judgment)
        qwn_failed = _failed_criteria(qwn_judgment)
        dsk_failed = _failed_criteria(dsk_judgment)
        ant_failed = _failed_criteria(ant_judgment)

        # Combine failed criteria for table display
        all_failed = []
        if oai_failed:
            all_failed.append(f"[cyan]OAI:[/cyan] {_fmt_failed(oai_failed)}")
        if gem_failed:
            all_failed.append(f"[magenta]GEM:[/magenta] {_fmt_failed(gem_failed)}")
        if qwn_failed:
            all_failed.append(f"[yellow]QWN:[/yellow] {_fmt_failed(qwn_failed)}")
        if dsk_failed:
            all_failed.append(f"[red]DSK:[/red] {_fmt_failed(dsk_failed)}")
        if ant_failed:
            all_failed.append(f"[green]ANT:[/green] {_fmt_failed(ant_failed)}")
        failed_display = "\n".join(all_failed) if all_failed else "[dim]—[/dim]"

        flow_short = flow.split(".")[-1].strip() if "." in flow else flow
        label = f"[bold]{flow_short}[/bold]\n{scenario}"

        table.add_row(
            case_id,
            label,
            "[green]✓[/green]" if oai_ok else "[red]✗[/red]",
            "[green]✓[/green]" if gem_ok else "[red]✗[/red]",
            "[green]✓[/green]" if qwn_ok else "[red]✗[/red]",
            "[green]✓[/green]" if dsk_ok else "[red]✗[/red]",
            "[green]✓[/green]" if ant_ok else "[red]✗[/red]",
            failed_display,
        )

        plain_rows.append({
            "id": case_id,
            "flow": flow,
            "scenario": scenario,
            "oai_ok": oai_ok,
            "gem_ok": gem_ok,
            "qwn_ok": qwn_ok,
            "dsk_ok": dsk_ok,
            "ant_ok": ant_ok,
            "oai_failed": oai_failed,
            "gem_failed": gem_failed,
            "qwn_failed": qwn_failed,
            "dsk_failed": dsk_failed,
            "ant_failed": ant_failed,
            "conversation": conversation,
            "expected_behaviors": expected_behaviors,
            "oai_response": oai_response,
            "gem_response": gem_response,
            "qwn_response": qwn_response,
            "dsk_response": dsk_response,
            "ant_response": ant_response,
            "oai_criteria": oai_judgment.get("criteria", []),
            "gem_criteria": gem_judgment.get("criteria", []),
            "qwn_criteria": qwn_judgment.get("criteria", []),
            "dsk_criteria": dsk_judgment.get("criteria", []),
            "ant_criteria": ant_judgment.get("criteria", []),
        })

        status = f"OAI: {'✓' if oai_ok else '✗'}  GEM: {'✓' if gem_ok else '✗'}  QWN: {'✓' if qwn_ok else '✗'}  DSK: {'✓' if dsk_ok else '✗'}  ANT: {'✓' if ant_ok else '✗'}"
        console.print(status)

    console.print()
    console.print(table)
    console.print(
        f"\n[bold]Results:[/bold]  "
        f"OpenAI: [cyan]{oai_pass}/{total}[/cyan] ({oai_pass/total*100:.1f}%)  |  "
        f"Gemini: [magenta]{gem_pass}/{total}[/magenta] ({gem_pass/total*100:.1f}%)  |  "
        f"Qwen: [yellow]{qwn_pass}/{total}[/yellow] ({qwn_pass/total*100:.1f}%)  |  "
        f"DeepSeek: [blue]{dsk_pass}/{total}[/blue] ({dsk_pass/total*100:.1f}%)  |  "
        f"Anthropic: [green]{ant_pass}/{total}[/green] ({ant_pass/total*100:.1f}%)"
    )

    if output_file:
        _write_txt(output_file, plain_rows, total, oai_pass, gem_pass, qwn_pass, dsk_pass, ant_pass)
        console.print(f"Saved results to [green]{output_file}[/green]")

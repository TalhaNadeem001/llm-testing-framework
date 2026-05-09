import asyncio
import json
import re
from datetime import datetime, timezone

from rich.console import Console
from rich.table import Table
from rich import box

from llm_tester.providers import call_openai, call_gemini, call_qwen, call_deepseek

JSON_INSTRUCTION = (
    '\n\nAlways respond with ONLY valid JSON, no extra text:\n'
    '{"intent": "<intent_name>", "confidence": "high|low", "reasoning": "<why>", "notes": "<optional>"}'
)


def _parse_intent_json(text: str) -> dict | None:
    """Extract JSON from model response, handling markdown code blocks."""
    if not text:
        return None
    # Strip markdown code fences
    clean = re.sub(r"```(?:json)?\s*|\s*```", "", text).strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        # Try to find first {...} block
        match = re.search(r"\{.*?\}", clean, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return None


def _check(parsed: dict | None, expected_intent: str, expected_confidence: str | None, check_confidence: bool) -> bool:
    if parsed is None:
        return False
    intent_ok = parsed.get("intent", "").strip().lower() == expected_intent.strip().lower()
    if not intent_ok:
        return False
    if check_confidence and expected_confidence:
        return parsed.get("confidence", "").strip().lower() == expected_confidence.strip().lower()
    return True


def _result_str(parsed: dict | None) -> str:
    if parsed is None:
        return "[red]PARSE_ERROR[/red]"
    intent = parsed.get("intent", "?")
    conf = parsed.get("confidence", "?")
    return f"{intent} ({conf})"


def _result_str_plain(parsed: dict | None) -> str:
    if parsed is None:
        return "PARSE_ERROR"
    intent = parsed.get("intent", "?")
    conf = parsed.get("confidence", "?")
    return f"{intent} ({conf})"


def _write_txt(output_file: str, cases: list[dict], total: int, oai_pass: int, gem_pass: int, qwn_pass: int, dsk_pass: int) -> None:
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines = [
        f"Batch Results — {timestamp}",
        "=" * 42,
        "",
    ]
    for c in cases:
        oai_tick = "✓" if c["oai_ok"] else "✗"
        gem_tick = "✓" if c["gem_ok"] else "✗"
        qwn_tick = "✓" if c["qwn_ok"] else "✗"
        dsk_tick = "✓" if c["dsk_ok"] else "✗"
        lines.append(f"[{c['id']}]  OAI: {oai_tick}  GEM: {gem_tick}  QWN: {qwn_tick}  DSK: {dsk_tick}")
        lines.append(f"Message : {c['message']}")
        lines.append(f"Expected: {c['expected_label']}")
        lines.append(f"OpenAI  : {c['oai_result']}")
        if c["oai_reasoning"]:
            lines.append(f"  Reasoning: {c['oai_reasoning']}")
        lines.append(f"Gemini  : {c['gem_result']}")
        if c["gem_reasoning"]:
            lines.append(f"  Reasoning: {c['gem_reasoning']}")
        lines.append(f"Qwen    : {c['qwn_result']}")
        if c["qwn_reasoning"]:
            lines.append(f"  Reasoning: {c['qwn_reasoning']}")
        lines.append(f"DeepSeek: {c['dsk_result']}")
        if c["dsk_reasoning"]:
            lines.append(f"  Reasoning: {c['dsk_reasoning']}")
        lines.append("")
    lines += [
        "-" * 42,
        f"OpenAI: {oai_pass}/{total} ({oai_pass/total*100:.0f}%)  |  Gemini: {gem_pass}/{total} ({gem_pass/total*100:.0f}%)  |  Qwen: {qwn_pass}/{total} ({qwn_pass/total*100:.0f}%)  |  DeepSeek: {dsk_pass}/{total} ({dsk_pass/total*100:.0f}%)",
    ]

    with open(output_file, "w") as f:
        f.write("\n".join(lines) + "\n")


async def run_batch(
    test_cases: list[dict],
    openai_model: str,
    gemini_model: str,
    qwen_model: str,
    system_prompt: str,
    check_confidence: bool = False,
    output_file: str | None = None,
    deepseek_model: str = "deepseek-v4-flash",
) -> None:
    console = Console()
    augmented_prompt = system_prompt + JSON_INSTRUCTION

    table = Table(box=box.SIMPLE_HEAVY, show_lines=False)
    table.add_column("ID", style="bold cyan", no_wrap=True)
    table.add_column("Message", max_width=40)
    table.add_column("Expected", style="yellow", no_wrap=True)
    table.add_column("OpenAI", no_wrap=True)
    table.add_column("Gemini", no_wrap=True)
    table.add_column("Qwen", no_wrap=True)
    table.add_column("DeepSeek", no_wrap=True)
    table.add_column("OAI", justify="center", no_wrap=True)
    table.add_column("GEM", justify="center", no_wrap=True)
    table.add_column("QWN", justify="center", no_wrap=True)
    table.add_column("DSK", justify="center", no_wrap=True)

    oai_pass = 0
    gem_pass = 0
    qwn_pass = 0
    dsk_pass = 0
    plain_rows = []

    for case in test_cases:
        case_id = case.get("id", "?")
        message = case.get("message", "")
        expected_intent = case.get("expected_intent", "")
        expected_confidence = case.get("expected_confidence", "")
        messages = [{"role": "user", "content": message}]

        oai_raw, gem_raw, qwn_raw, dsk_raw = await asyncio.gather(
            asyncio.to_thread(call_openai, messages, openai_model, augmented_prompt),
            asyncio.to_thread(call_gemini, messages, gemini_model, augmented_prompt),
            asyncio.to_thread(call_qwen, messages, qwen_model, augmented_prompt),
            asyncio.to_thread(call_deepseek, messages, deepseek_model, augmented_prompt),
        )

        oai_parsed = _parse_intent_json(oai_raw)
        gem_parsed = _parse_intent_json(gem_raw)
        qwn_parsed = _parse_intent_json(qwn_raw)
        dsk_parsed = _parse_intent_json(dsk_raw)

        oai_ok = _check(oai_parsed, expected_intent, expected_confidence, check_confidence)
        gem_ok = _check(gem_parsed, expected_intent, expected_confidence, check_confidence)
        qwn_ok = _check(qwn_parsed, expected_intent, expected_confidence, check_confidence)
        dsk_ok = _check(dsk_parsed, expected_intent, expected_confidence, check_confidence)

        if oai_ok:
            oai_pass += 1
        if gem_ok:
            gem_pass += 1
        if qwn_ok:
            qwn_pass += 1
        if dsk_ok:
            dsk_pass += 1

        expected_label = expected_intent
        if check_confidence and expected_confidence:
            expected_label += f" ({expected_confidence})"

        short_msg = message if len(message) <= 40 else message[:37] + "..."
        table.add_row(
            case_id,
            short_msg,
            expected_label,
            _result_str(oai_parsed),
            _result_str(gem_parsed),
            _result_str(qwn_parsed),
            _result_str(dsk_parsed),
            "[green]✓[/green]" if oai_ok else "[red]✗[/red]",
            "[green]✓[/green]" if gem_ok else "[red]✗[/red]",
            "[green]✓[/green]" if qwn_ok else "[red]✗[/red]",
            "[green]✓[/green]" if dsk_ok else "[red]✗[/red]",
        )
        plain_rows.append({
            "id": case_id,
            "message": message,
            "expected_label": expected_label,
            "oai_result": _result_str_plain(oai_parsed),
            "oai_reasoning": (oai_parsed or {}).get("reasoning", ""),
            "oai_ok": oai_ok,
            "gem_result": _result_str_plain(gem_parsed),
            "gem_reasoning": (gem_parsed or {}).get("reasoning", ""),
            "gem_ok": gem_ok,
            "qwn_result": _result_str_plain(qwn_parsed),
            "qwn_reasoning": (qwn_parsed or {}).get("reasoning", ""),
            "qwn_ok": qwn_ok,
            "dsk_result": _result_str_plain(dsk_parsed),
            "dsk_reasoning": (dsk_parsed or {}).get("reasoning", ""),
            "dsk_ok": dsk_ok,
        })

    total = len(test_cases)
    console.print(table)
    console.print(
        f"[bold]Results:[/bold]  "
        f"OpenAI: [cyan]{oai_pass}/{total}[/cyan] ({oai_pass/total*100:.0f}%)  |  "
        f"Gemini: [magenta]{gem_pass}/{total}[/magenta] ({gem_pass/total*100:.0f}%)  |  "
        f"Qwen: [yellow]{qwn_pass}/{total}[/yellow] ({qwn_pass/total*100:.0f}%)  |  "
        f"DeepSeek: [blue]{dsk_pass}/{total}[/blue] ({dsk_pass/total*100:.0f}%)"
    )

    if output_file:
        _write_txt(output_file, plain_rows, total, oai_pass, gem_pass, qwn_pass, dsk_pass)
        console.print(f"Saved results to {output_file}")

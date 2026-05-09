import asyncio
import importlib.util
import json
import os
from datetime import datetime, timezone

from rich.console import Console
from rich.table import Table
from rich import box
from openai import OpenAI
from google import genai
from google.genai import types

_tools_spec = importlib.util.spec_from_file_location("agent_tools", "Test Execution agent/tools.py")
_tools_mod = importlib.util.module_from_spec(_tools_spec)
_tools_spec.loader.exec_module(_tools_mod)
TOOL_DEFINITIONS = _tools_mod.TOOL_DEFINITIONS

MAX_TURNS = 5


def _load_tool_stubs():
    """Import tool stub functions from 'Test Execution agent/tools.py'."""
    spec = importlib.util.spec_from_file_location("agent_tools", "Test Execution agent/tools.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _execute_tool(stubs_mod, name: str, args: dict) -> str:
    fn = getattr(stubs_mod, name, None)
    if fn is None:
        return json.dumps({"error": f"unknown tool: {name}"})
    try:
        result = fn(**args)
    except Exception as e:
        result = {"error": str(e)}
    return json.dumps(result)


def _call_openai_tools(message: str, model: str, system_prompt: str) -> list[str]:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    stubs = _load_tool_stubs()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": message},
    ]
    all_called = []

    for _ in range(MAX_TURNS):
        response = client.chat.completions.create(
            model=model, messages=messages, tools=TOOL_DEFINITIONS, tool_choice="auto"
        )
        msg = response.choices[0].message
        tool_calls = msg.tool_calls or []
        if not tool_calls:
            break

        all_called.extend(tc.function.name for tc in tool_calls)
        messages.append(msg)

        for tc in tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            result = _execute_tool(stubs, tc.function.name, args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    return all_called


def _call_deepseek_tools(message: str, model: str, system_prompt: str) -> list[str]:
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
    )
    stubs = _load_tool_stubs()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": message},
    ]
    all_called = []

    for _ in range(MAX_TURNS):
        response = client.chat.completions.create(
            model=model, messages=messages, tools=TOOL_DEFINITIONS, tool_choice="auto"
        )
        msg = response.choices[0].message
        tool_calls = msg.tool_calls or []
        if not tool_calls:
            break

        all_called.extend(tc.function.name for tc in tool_calls)
        messages.append(msg)

        for tc in tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            result = _execute_tool(stubs, tc.function.name, args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    return all_called


def _call_qwen_tools(message: str, model: str, system_prompt: str) -> list[str]:
    client = OpenAI(
        api_key=os.environ["DASHSCOPE_API_KEY"],
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    )
    stubs = _load_tool_stubs()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": message},
    ]
    all_called = []

    for _ in range(MAX_TURNS):
        response = client.chat.completions.create(
            model=model, messages=messages, tools=TOOL_DEFINITIONS, tool_choice="auto"
        )
        msg = response.choices[0].message
        tool_calls = msg.tool_calls or []
        if not tool_calls:
            break

        all_called.extend(tc.function.name for tc in tool_calls)
        messages.append(msg)

        for tc in tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            result = _execute_tool(stubs, tc.function.name, args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    return all_called


def _build_gemini_tools() -> list:
    declarations = []
    for td in TOOL_DEFINITIONS:
        fn = td["function"]
        params = fn.get("parameters", {})
        props = params.get("properties", {})
        required = params.get("required", [])

        gemini_props = {}
        for name, schema in props.items():
            t = schema.get("type", "string")
            type_map = {
                "string": types.Type.STRING,
                "integer": types.Type.INTEGER,
                "boolean": types.Type.BOOLEAN,
                "number": types.Type.NUMBER,
            }
            gemini_props[name] = types.Schema(type=type_map.get(t, types.Type.STRING))

        parameters = types.Schema(
            type=types.Type.OBJECT,
            properties=gemini_props,
            required=required,
        ) if gemini_props else None

        decl_kwargs = {
            "name": fn["name"],
            "description": fn.get("description", ""),
        }
        if parameters is not None:
            decl_kwargs["parameters"] = parameters

        declarations.append(types.FunctionDeclaration(**decl_kwargs))

    return [types.Tool(function_declarations=declarations)]


def _call_gemini_tools(message: str, model: str, system_prompt: str) -> list[str]:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    stubs = _load_tool_stubs()
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        tools=_build_gemini_tools(),
    )
    contents = [types.Content(role="user", parts=[types.Part(text=message)])]
    all_called = []

    for _ in range(MAX_TURNS):
        response = client.models.generate_content(model=model, contents=contents, config=config)
        candidate = response.candidates[0]
        fn_calls = [p.function_call for p in candidate.content.parts if p.function_call]
        if not fn_calls:
            break

        all_called.extend(fc.name for fc in fn_calls)
        contents.append(candidate.content)

        result_parts = []
        for fc in fn_calls:
            args = dict(fc.args)
            result = _execute_tool(stubs, fc.name, args)
            result_parts.append(types.Part(
                function_response=types.FunctionResponse(
                    name=fc.name,
                    response={"result": result},
                )
            ))
        contents.append(types.Content(role="user", parts=result_parts))

    return all_called


def _check(called: list[str], expected: list[str]) -> bool:
    return set(called) == set(expected)


def _fmt_tools(tools: list[str]) -> str:
    return ", ".join(tools) if tools else "(none)"


def _write_txt(
    output_file: str,
    cases: list[dict],
    total: int,
    oai_pass: int,
    gem_pass: int,
    qwn_pass: int,
    dsk_pass: int,
) -> None:
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines = [
        f"Execution Batch Results — {timestamp}",
        "=" * 52,
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
        lines.append(f"Gemini  : {c['gem_result']}")
        lines.append(f"Qwen    : {c['qwn_result']}")
        lines.append(f"DeepSeek: {c['dsk_result']}")
        lines.append("")
    lines += [
        "-" * 52,
        f"OpenAI: {oai_pass}/{total} ({oai_pass/total*100:.0f}%)  |  Gemini: {gem_pass}/{total} ({gem_pass/total*100:.0f}%)  |  Qwen: {qwn_pass}/{total} ({qwn_pass/total*100:.0f}%)  |  DeepSeek: {dsk_pass}/{total} ({dsk_pass/total*100:.0f}%)",
    ]
    with open(output_file, "w") as f:
        f.write("\n".join(lines) + "\n")


async def run_execution_batch(
    test_cases: list[dict],
    openai_model: str,
    gemini_model: str,
    system_prompt: str,
    output_file: str | None = None,
    qwen_model: str = "qwen3.5-flash",
    deepseek_model: str = "deepseek-v4-flash",
) -> None:
    console = Console()

    table = Table(box=box.SIMPLE_HEAVY, show_lines=False)
    table.add_column("ID", style="bold cyan", no_wrap=True)
    table.add_column("Message", max_width=38)
    table.add_column("Expected Tools", max_width=30)
    table.add_column("OpenAI Called", max_width=30)
    table.add_column("Gemini Called", max_width=30)
    table.add_column("Qwen Called", max_width=30)
    table.add_column("DeepSeek Called", max_width=30)
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
        expected_tools = case.get("expected_tools", [])

        oai_called, gem_called, qwn_called, dsk_called = await asyncio.gather(
            asyncio.to_thread(_call_openai_tools, message, openai_model, system_prompt),
            asyncio.to_thread(_call_gemini_tools, message, gemini_model, system_prompt),
            asyncio.to_thread(_call_qwen_tools, message, qwen_model, system_prompt),
            asyncio.to_thread(_call_deepseek_tools, message, deepseek_model, system_prompt),
        )

        oai_ok = _check(oai_called, expected_tools)
        gem_ok = _check(gem_called, expected_tools)
        qwn_ok = _check(qwn_called, expected_tools)
        dsk_ok = _check(dsk_called, expected_tools)

        if oai_ok:
            oai_pass += 1
        if gem_ok:
            gem_pass += 1
        if qwn_ok:
            qwn_pass += 1
        if dsk_ok:
            dsk_pass += 1

        expected_label = _fmt_tools(expected_tools)
        oai_result = _fmt_tools(oai_called)
        gem_result = _fmt_tools(gem_called)
        qwn_result = _fmt_tools(qwn_called)
        dsk_result = _fmt_tools(dsk_called)

        short_msg = message if len(message) <= 38 else message[:35] + "..."
        table.add_row(
            case_id,
            short_msg,
            expected_label,
            oai_result,
            gem_result,
            qwn_result,
            dsk_result,
            "[green]✓[/green]" if oai_ok else "[red]✗[/red]",
            "[green]✓[/green]" if gem_ok else "[red]✗[/red]",
            "[green]✓[/green]" if qwn_ok else "[red]✗[/red]",
            "[green]✓[/green]" if dsk_ok else "[red]✗[/red]",
        )
        plain_rows.append({
            "id": case_id,
            "message": message,
            "expected_label": expected_label,
            "oai_result": oai_result,
            "oai_ok": oai_ok,
            "gem_result": gem_result,
            "gem_ok": gem_ok,
            "qwn_result": qwn_result,
            "qwn_ok": qwn_ok,
            "dsk_result": dsk_result,
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

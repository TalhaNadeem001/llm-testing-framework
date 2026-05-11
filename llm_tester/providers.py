import os
import anthropic
from openai import OpenAI
from google import genai
from google.genai import types


def call_openai(messages: list[dict], model: str = "gpt-5.4-mini", system_prompt: str = "") -> str:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    full_messages = []
    if system_prompt:
        full_messages.append({"role": "system", "content": system_prompt})
    full_messages.extend(messages)
    response = client.chat.completions.create(model=model, messages=full_messages)
    return response.choices[0].message.content


def call_gemini(messages: list[dict], model: str = "gemini-3-flash-preview", system_prompt: str = "") -> str:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    config = types.GenerateContentConfig(system_instruction=system_prompt) if system_prompt else None

    contents = []
    for msg in messages:
        role = "model" if msg["role"] == "assistant" else msg["role"]
        contents.append(types.Content(role=role, parts=[types.Part(text=msg["content"])]))

    response = client.models.generate_content(model=model, contents=contents, config=config)
    return response.text


def call_qwen(messages: list[dict], model: str = "qwen3.5-flash", system_prompt: str = "") -> str:
    client = OpenAI(
        api_key=os.environ["DASHSCOPE_API_KEY"],
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    )
    full_messages = []
    if system_prompt:
        full_messages.append({"role": "system", "content": system_prompt})
    full_messages.extend(messages)
    response = client.chat.completions.create(model=model, messages=full_messages)
    return response.choices[0].message.content


def call_anthropic(messages: list[dict], model: str = "claude-sonnet-4-6", system_prompt: str = "") -> str:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    kwargs = {"model": model, "max_tokens": 1024, "messages": messages}
    if system_prompt:
        kwargs["system"] = system_prompt
    response = client.messages.create(**kwargs)
    return response.content[0].text


def call_deepseek(messages: list[dict], model: str = "deepseek-v4-flash", system_prompt: str = "") -> str:
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
    )
    full_messages = []
    if system_prompt:
        full_messages.append({"role": "system", "content": system_prompt})
    full_messages.extend(messages)
    response = client.chat.completions.create(model=model, messages=full_messages)
    return response.choices[0].message.content

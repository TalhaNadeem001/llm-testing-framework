import os
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

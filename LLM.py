# ollama_client.py

import requests

OLLAMA_HOST  = "http://127.0.0.1:11434"
OLLAMA_MODEL = "deepseek-r1"  # Или любая установленная модель через `ollama run`

def strip_think_sections(raw: str) -> str:
    parts = raw.split("</think>")
    if len(parts) >= 3:
        return parts[2].strip()
    elif len(parts) == 2:
        return parts[1].strip()
    else:
        return raw.strip()

def ask_ollama_http(prompt: str, max_tokens: int = 2048, temperature: float = 0.0) -> str:
    instruction = "Дай максимально лаконичный ответ на данный вопрос.\n\n"
    payload_prompt = instruction + prompt

    url = f"{OLLAMA_HOST}/v1/completions"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": payload_prompt,
        "max_tokens": max_tokens,
        "temperature": temperature
    }

    resp = requests.post(url, json=payload)
    resp.raise_for_status()
    data = resp.json()

    if "completion" in data:
        return strip_think_sections(data["completion"])
    else:
        return str(data)

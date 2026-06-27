import os
import json
import logging

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
VISION_MODEL = os.environ.get("VISION_MODEL", "meta-llama/llama-4-maverick:free")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Особистий застосунок без реєстрації -> дозволяємо запити з будь-якого джерела
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SYSTEM_PROMPT = """Ти — нутриціолог-асистент. Тобі надсилають фото їжі та/або текстовий опис того,
що людина з'їла. Проаналізуй це і поверни ВИКЛЮЧНО валідний JSON (без markdown, без пояснень,
без ```), що відповідає рівно такій схемі:

{
  "name": "коротка назва страви українською",
  "calories_min": число,
  "calories_max": число,
  "protein_g": число,
  "fat_g": число,
  "carbs_g": число,
  "recommendation": "1-2 речення українською: чи варто щось замінити (на що і чому) чи можна лишити як є. Дружній тон, без присоромлення.",
  "confidence": "high" | "medium" | "low"
}

Якщо фото нечітке або опис надто короткий — все одно дай орієнтовну оцінку, але поставь
confidence "low" і у recommendation вкажи, що оцінка приблизна."""


class AnalyzeRequest(BaseModel):
    text: Optional[str] = None
    image_base64: Optional[str] = None


def extract_json(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
    start = raw.find("{")
    end = raw.rfind("}")
    raw = raw[start : end + 1]
    return json.loads(raw)


@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    user_content = []
    text_part = req.text or "Проаналізуй цю їжу."
    user_content.append({"type": "text", "text": text_part})

    if req.image_base64:
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{req.image_base64}"},
            }
        )

    payload = {
        "model": VISION_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": 500,
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(OPENROUTER_URL, headers=headers, json=payload)
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]

    try:
        parsed = extract_json(raw)
    except Exception:
        logger.warning("Не вдалося розпарсити JSON від моделі: %s", raw)
        parsed = {
            "name": "Невідома страва",
            "calories_min": None,
            "calories_max": None,
            "protein_g": None,
            "fat_g": None,
            "carbs_g": None,
            "recommendation": raw[:300],
            "confidence": "low",
        }

    return parsed


@app.get("/health")
async def health():
    return {"status": "ok"}

import json
import os
import re
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from groq import Groq

BOT_TOKEN = os.environ["BOT_TOKEN"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
PUBLIC_BASE_URL = os.environ["PUBLIC_BASE_URL"].rstrip("/")
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "")
MODEL = os.environ.get("GROQ_MODEL", "groq/compound")

TELEGRAM_API = f"https://api.telegram.org/bot{BOT_TOKEN}"
LOG_DIR = Path(os.environ.get("LOG_DIR", "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
LATEST_LOG = LOG_DIR / "run.jsonl"

app = FastAPI(title="Data Analyst Telegram Bot")
groq = Groq(api_key=GROQ_API_KEY)

# Keep recent conversation turns per Telegram chat for multi-turn questions.
chat_history: dict[int, deque[dict[str, str]]] = defaultdict(
    lambda: deque(maxlen=4)
)

SYSTEM_PROMPT = """
You are a rigorous data-analysis agent.

You receive Telegram messages containing data-analysis questions. The data may
be inline or available through a public URL, including official government
datasets such as MOSPI.

Rules:
1. Solve the user's latest question accurately.
2. Use web search, website visiting, code execution, and calculations whenever useful.
3. Respect the complete recent conversation because questions may be multi-turn.
4. The user's latest message usually states the exact required JSON shape.
5. Return only one valid JSON object with exactly one top-level key: "answer". Never repeat or copy the user's question or placeholder template.
6. Put inside "answer" precisely the value/object/list requested by the user.
7. Do not include markdown, explanations, citations, code fences, or a log_url.
8. Preserve requested spelling, data types, rounding, ordering, and key names.
9. Never invent data. If a public source is required, find and analyze it.
"""

def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_jsonl(events: list[dict[str, Any]]) -> None:
    """Replace the public log with the latest run, one JSON object per line."""
    temp = LATEST_LOG.with_suffix(".tmp")
    with temp.open("w", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
    temp.replace(LATEST_LOG)


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if hasattr(value, "model_dump"):
        return jsonable(value.model_dump())
    if hasattr(value, "to_dict"):
        return jsonable(value.to_dict())
    return str(value)


def extract_json_object(text: str) -> dict[str, Any]:
    text = (text or "").strip()

    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass

    # Fallback: locate the first balanced JSON object.
    start = text.find("{")
    if start == -1:
        raise ValueError("Model returned no JSON object.")

    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                value = json.loads(text[start : i + 1])
                if not isinstance(value, dict):
                    raise ValueError("JSON result was not an object.")
                return value

    raise ValueError("Could not parse a complete JSON object.")


def normalize_answer_object(model_object: dict[str, Any]) -> Any:
    """
    The model is instructed to return {"answer": ...}. This fallback also
    handles a model that directly returns the requested answer object.
    """
    if set(model_object.keys()) == {"answer"}:
        return model_object["answer"]
    return model_object


def solve_question(chat_id: int, latest_text: str) -> tuple[Any, list[dict[str, Any]]]:
    run_id = str(uuid.uuid4())
    events: list[dict[str, Any]] = [
        {
            "event": "run_started",
            "run_id": run_id,
            "timestamp": now_iso(),
            "chat_id": chat_id,
            "latest_message": latest_text,
        }
    ]

    history = list(chat_history[chat_id])[-4:]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append(
    {
        "role": "user",
        "content": f"""
        Analyze the question below and give the real answer.

        Do not repeat the question.
        Do not copy placeholder values such as "<state name>".
        Return only this internal JSON format:

        {{"answer": {{"state": "actual state name"}}}}
        
        QUESTION:
        {latest_text[:12000]}
"""
    }
    )

    events.append(
        {
            "event": "llm_request",
            "run_id": run_id,
            "timestamp": now_iso(),
            "model": MODEL,
            "messages": messages,
            "enabled_tools": [
                "web_search",
                "visit_website",
                "code_interpreter",
            ],
        }
    )

    completion = groq.chat.completions.create(
    model=MODEL,
    messages=messages,
    temperature=0,
    max_completion_tokens=2048,
    )

    msg = completion.choices[0].message
    raw_text = msg.content or ""
    executed_tools = jsonable(getattr(msg, "executed_tools", None))
    reasoning = jsonable(getattr(msg, "reasoning", None))

    events.append(
        {
            "event": "llm_response",
            "run_id": run_id,
            "timestamp": now_iso(),
            "raw_content": raw_text,
            "executed_tools": executed_tools,
            "reasoning": reasoning,
            "usage": jsonable(getattr(completion, "usage", None)),
        }
    )

    parsed = extract_json_object(raw_text)
    raw_lower = raw_text.lower()
    invalid_placeholders = [
        "<state name>",
        "<public wget-able url>",
        "<url>",
        "actual state name",
        ]
    if any(value in raw_lower for value in invalid_placeholders):
        raise ValueError("Model copied the JSON template instead of answering.")

    answer = normalize_answer_object(parsed)

    events.append(
        {
            "event": "answer_validated",
            "run_id": run_id,
            "timestamp": now_iso(),
            "answer": answer,
        }
    )

    # Save only user/assistant semantic context, not the public log URL.
    chat_history[chat_id].append({"role": "user", "content": latest_text})
    chat_history[chat_id].append(
        {
            "role": "assistant",
            "content": json.dumps({"answer": answer}, ensure_ascii=False),
        }
    )

    return answer, events


async def send_telegram_json(chat_id: int, payload: dict[str, Any]) -> None:
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            f"{TELEGRAM_API}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": text,
                "disable_web_page_preview": True,
            },
        )
    response.raise_for_status()


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "ok", "service": "data-analyst-telegram-bot"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/run.jsonl")
def public_run_log():
    if not LATEST_LOG.exists():
        return PlainTextResponse(
            json.dumps(
                {
                    "event": "no_runs_yet",
                    "timestamp": now_iso(),
                }
            )
            + "\n",
            media_type="application/x-ndjson",
        )
    return FileResponse(
        LATEST_LOG,
        media_type="application/x-ndjson",
        filename="run.jsonl",
    )


@app.post("/telegram/webhook")
async def telegram_webhook(request: Request):
    if WEBHOOK_SECRET:
        received = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
        if received != WEBHOOK_SECRET:
            raise HTTPException(status_code=403, detail="Invalid webhook secret")

    update = await request.json()
    message = update.get("message") or update.get("edited_message")
    if not message:
        return JSONResponse({"ok": True})

    text = message.get("text")
    chat = message.get("chat") or {}
    chat_id = chat.get("id")

    if not isinstance(text, str) or not isinstance(chat_id, int):
        return JSONResponse({"ok": True})

    if text.strip().lower() == "/reset":
        chat_history.pop(chat_id, None)

    await send_telegram_json(
        chat_id,
        {
            "answer": {"status": "conversation_reset"},
            "log_url": f"{PUBLIC_BASE_URL}/run.jsonl",
        },
    )

    return JSONResponse({"ok": True})

    # Telegram retries unsuccessful webhooks. Return 200 even if analysis fails,
    # after sending a valid JSON failure response.
    
    try:
        answer, events = solve_question(chat_id, text)
        final_payload = {
            "answer": answer,
            "log_url": f"{PUBLIC_BASE_URL}/run.jsonl",
        }
        events.append(
            {
                "event": "telegram_reply",
                "timestamp": now_iso(),
                "payload": final_payload,
            }
        )
        write_jsonl(events)
        await send_telegram_json(chat_id, final_payload)
    except Exception as exc:
        failure_events = [
            {
                "event": "run_failed",
                "timestamp": now_iso(),
                "chat_id": chat_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        ]
        write_jsonl(failure_events)
        # Still obey the required two-key JSON envelope.
        await send_telegram_json(
            chat_id,
            {
                "answer": {"error": "analysis_failed"},
                "log_url": f"{PUBLIC_BASE_URL}/run.jsonl",
            },
        )

    return JSONResponse({"ok": True})


@app.post("/admin/set-webhook")
async def set_webhook(request: Request):
    supplied = request.headers.get("X-Admin-Secret", "")
    admin_secret = os.environ.get("ADMIN_SECRET", "")
    if not admin_secret or supplied != admin_secret:
        raise HTTPException(status_code=403, detail="Forbidden")

    webhook_url = f"{PUBLIC_BASE_URL}/telegram/webhook"
    body: dict[str, Any] = {
        "url": webhook_url,
        "allowed_updates": ["message", "edited_message"],
        "drop_pending_updates": True,
    }
    if WEBHOOK_SECRET:
        body["secret_token"] = WEBHOOK_SECRET

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(f"{TELEGRAM_API}/setWebhook", json=body)
    return JSONResponse(response.json(), status_code=response.status_code)

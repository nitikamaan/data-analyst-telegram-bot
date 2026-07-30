import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
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

SYSTEM_PROMPT = """
You are a rigorous data-analysis agent.

Solve the user's data-analysis question accurately.

Rules:
1. Use calculations, web research, official public datasets, and code execution
   whenever required.
2. For government-data questions, prefer official government sources.
3. Answer the actual question. Never repeat the question.
4. Never copy placeholder values such as <state name>, <number>, or <URL>.
5. Return exactly one valid JSON object with exactly one top-level key: "answer".
6. Put the actual requested result inside "answer".
7. Do not include markdown, code fences, explanations, citations, or log_url.
8. Preserve the requested key names, value types, spelling, and rounding.
9. Never invent data.

Examples:

User asks for a mean:
{"answer":{"mean":20}}

User asks for a state:
{"answer":{"state":"Assam"}}
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
    """
    Parse the model response and prefer the last valid JSON object whose
    top-level key is exactly "answer". This avoids accidentally extracting
    a copied template from earlier in the response.
    """
    cleaned = (text or "").strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.replace("```json", "", 1).replace("```", "", 1).strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()

    try:
        value = json.loads(cleaned)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass

    objects: list[dict[str, Any]] = []
    start_index: int | None = None
    depth = 0
    in_string = False
    escaped = False

    for index, ch in enumerate(cleaned):
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
            if depth == 0:
                start_index = index
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start_index is not None:
                candidate = cleaned[start_index:index + 1]
                try:
                    value = json.loads(candidate)
                    if isinstance(value, dict):
                        objects.append(value)
                except json.JSONDecodeError:
                    pass
                start_index = None

    if not objects:
        raise ValueError("Model returned no valid JSON object.")

    for value in reversed(objects):
        if set(value.keys()) == {"answer"}:
            return value

    return objects[-1]

def normalize_answer_object(model_object: dict[str, Any]) -> Any:
    """
    The model is instructed to return {"answer": ...}. This fallback also
    handles a model that directly returns the requested answer object.
    """
    if set(model_object.keys()) == {"answer"}:
        return model_object["answer"]
    return model_object


def solve_question(
    chat_id: int,
    latest_text: str,
) -> tuple[Any, list[dict[str, Any]]]:

    run_id = str(uuid.uuid4())

    # Limit the current question size.
    safe_text = latest_text[:4000]

    events: list[dict[str, Any]] = [
        {
            "event": "run_started",
            "run_id": run_id,
            "timestamp": now_iso(),
            "chat_id": chat_id,
            "message_length": len(latest_text),
            "message_preview": safe_text[:500],
        }
    ]

    # Do not include previous chat history.
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": f"""
Solve the following question.

Return the real answer, not the placeholder template.
Do not repeat the question.
Return only one JSON object with the top-level key "answer".

QUESTION:
{safe_text}
""".strip(),
        },
    ]

    events.append(
        {
            "event": "llm_request",
            "run_id": run_id,
            "timestamp": now_iso(),
            "model": MODEL,
            "message_count": len(messages),
            "prompt_character_count": sum(
                len(message["content"])
                for message in messages
            ),
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
        max_completion_tokens=1024,
    )

    if not completion.choices:
        raise ValueError("Groq returned no completion choices.")

    msg = completion.choices[0].message
    raw_text = msg.content or ""

    if not raw_text.strip():
        raise ValueError("Groq returned an empty response.")

    executed_tools = jsonable(
        getattr(msg, "executed_tools", None)
    )

    reasoning = jsonable(
        getattr(msg, "reasoning", None)
    )

    events.append(
        {
            "event": "llm_response",
            "run_id": run_id,
            "timestamp": now_iso(),
            "raw_content": raw_text[:6000],
            "executed_tools": executed_tools,
            "reasoning": reasoning,
            "usage": jsonable(
                getattr(completion, "usage", None)
            ),
        }
    )

    parsed = extract_json_object(raw_text)

    answer = normalize_answer_object(parsed)

    if answer is None:
        raise ValueError("Model returned a null answer.")

    answer_text = json.dumps(answer, ensure_ascii=False).lower()
    invalid_placeholders = [
        "<state name>",
        "<number>",
        "<value>",
        "<answer>",
        "<public wget-able url>",
        "<public url>",
        "<url>",
        "actual state name",
    ]

    if any(placeholder in answer_text for placeholder in invalid_placeholders):
        raise ValueError(
            "Model copied a placeholder instead of answering."
        )

    events.append(
        {
            "event": "answer_validated",
            "run_id": run_id,
            "timestamp": now_iso(),
            "answer": answer,
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
        received_secret = request.headers.get(
            "X-Telegram-Bot-Api-Secret-Token",
            "",
        )

        if received_secret != WEBHOOK_SECRET:
            raise HTTPException(
                status_code=403,
                detail="Invalid webhook secret",
            )

    update = await request.json()

    message = update.get("message") or update.get("edited_message")

    if not message:
        return JSONResponse({"ok": True})

    text = message.get("text")
    chat = message.get("chat") or {}
    chat_id = chat.get("id")

    if not isinstance(text, str) or not isinstance(chat_id, int):
        return JSONResponse({"ok": True})

    text = text.strip()

    if not text:
        return JSONResponse({"ok": True})

    # Reset ONLY when the received message is exactly /reset
    first_word = text.split()[0].lower()

    if first_word == "/reset" or first_word.startswith("/reset@"):
        reset_payload = {
            "answer": {
                "status": "conversation_reset"
            },
            "log_url": f"{PUBLIC_BASE_URL}/run.jsonl",
        }

        write_jsonl(
            [
                {
                    "event": "conversation_reset",
                    "timestamp": now_iso(),
                    "chat_id": chat_id,
                    "received_text": text,
                }
            ]
        )

        await send_telegram_json(chat_id, reset_payload)

        return JSONResponse({"ok": True})

    # Every other message must reach this part
    try:
        answer, events = await run_in_threadpool(
            solve_question,
            chat_id,
            text,
        )

        final_payload = {
            "answer": answer,
            "log_url": f"{PUBLIC_BASE_URL}/run.jsonl",
        }

        events.append(
            {
                "event": "telegram_reply",
                "timestamp": now_iso(),
                "received_text": text,
                "payload": final_payload,
            }
        )

        write_jsonl(events)

        await send_telegram_json(
            chat_id,
            final_payload,
        )

    except Exception as exc:
        failure_payload = {
            "answer": {
                "error": "analysis_failed",
                "detail": str(exc)[:500],
            },
            "log_url": f"{PUBLIC_BASE_URL}/run.jsonl",
        }

        write_jsonl(
            [
                {
                    "event": "run_failed",
                    "timestamp": now_iso(),
                    "chat_id": chat_id,
                    "received_text": text,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            ]
        )

        await send_telegram_json(
            chat_id,
            failure_payload,
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
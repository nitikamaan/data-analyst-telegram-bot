# Data Analyst Telegram Bot

A Telegram webhook bot that solves data-analysis questions with Groq Compound,
returns exactly one JSON object, and exposes its latest JSONL run log publicly.

## Required environment variables

- `BOT_TOKEN`
- `GROQ_API_KEY`
- `PUBLIC_BASE_URL`
- `WEBHOOK_SECRET`
- `ADMIN_SECRET`

## Local run

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

## Public endpoints

- `GET /health`
- `GET /run.jsonl`
- `POST /telegram/webhook`
- `POST /admin/set-webhook`

## Set webhook

```bash
curl -X POST "https://YOUR-DOMAIN/admin/set-webhook"   -H "X-Admin-Secret: YOUR_ADMIN_SECRET"
```

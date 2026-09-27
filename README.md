# Magicpin Vera AI Bot

Functional project skeleton and HTTP API for Magicpin Vera AI challenge bot.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Application
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

## API Endpoints

- `POST /v1/context`: Ingest category, merchant, customer, or trigger contexts with versioning support.
- `POST /v1/tick`: Periodic evaluation endpoint for trigger-based actions.
- `POST /v1/reply`: Receive customer/merchant messages and record conversation state.
- `GET /v1/healthz`: Health status, uptime, and loaded context counts.
- `GET /v1/metadata`: Metadata containing team info, model, version, and submission details.

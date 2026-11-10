# Yami Intake Assistant

Research prototype for structured psychiatric pre-consultation history taking.
A shared conversation pipeline serves a FastAPI REST interface and a Telegram
bot, records structured intake fields and prepares a report for human review.
Speech input uses a local SraVaani adapter; model inference can use a local
Transformers checkpoint or a separately managed vLLM server.

This is an application showcase, not an autonomous therapist, diagnostic system
or emergency service. It has not been clinically validated for deployment.

## Included

- `api.py`, `bot.py`, `handlers.py`: REST and Telegram transports.
- `turn_pipeline.py`, `clinical_state.py`, `data_model.py`: conversation state
  and structured intake sections.
- `llm.py`, `translation.py`: generation, extraction and translation.
- `sravaani_stt/`, `stt.py`: speech segmentation and transcription.
- `db.py`, `report.py`: local persistence and report generation.
- `safety.py`, `crisis.py`: research safety/crisis handling logic.

No user database, conversations, recordings, API keys, Telegram tokens, private
Git history or live deployment configuration is included.

## Setup

Use Python 3.11 and a suitable PyTorch build:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Edit `.env`. Generate your own API key, for example with
`python -c "import secrets; print(secrets.token_urlsafe(32))"`.
The public API has no shared fallback key. Protected endpoints return 503 when
`API_KEY` is absent and 401 for an invalid key. `/health` and schema/docs endpoints
are public. Keep the service on loopback during local development.

## Start the API

Start a compatible local vLLM server separately and set its model name/base URL
in `.env`, or configure `LLM_BACKEND=hf` with an authorized local checkpoint.
Automatic server/GPU management is disabled in the public default.

```bash
python -m uvicorn api:app --host 127.0.0.1 --port 5000
```

Open `http://127.0.0.1:5000/docs` for the API schema. Requests to protected endpoints
need `Authorization: Bearer <your-key>`. The console clients also read `API_KEY`
from their environment; export it before invoking them. For Telegram, supply
`TELEGRAM_BOT_TOKEN` and run `python bot.py`. Do not run both transports against
real users without reviewing session isolation and your deployment controls.

For speech, supply a local SraVaani checkpoint through `SRAVAANI_MODEL_PATH`.
No Whisper ASR weights are loaded by this adapter. FFmpeg-compatible audio
support and the declared faster-whisper decoder/VAD dependency are required.

## Offline checks

```bash
python -m unittest test_public_auth test_stt -v
python -m compileall -q .
```

These use synthetic requests, a temporary SQLite database and mocked model
responses. `test_api.py` is an optional integration client for a running server;
it can write test sessions and call models, so use a disposable deployment.
Full model generation and real patient workflows were not tested for this release.

## Privacy and limitations

The default SQLite database and logs are local and ignored by Git. They can
contain highly sensitive content; shared-key authentication alone is not a
production privacy architecture. Add user-level authorization, retention rules,
audit review, transport security and clinical oversight before real use.
Extraction, translation and crisis detection can fail. The Mental State
Examination section is disabled because text-only conversation cannot establish
observational findings. A generated report requires clinician review.

## License

Original application code is MIT licensed. Model weights and third-party
libraries retain their own terms. See `THIRD_PARTY_NOTICES.md`.

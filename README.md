# Turkey Bot

A LINE chatbot powered by LLM APIs (Google Gemini or OpenAI-compatible).

## Features

- **Multi-API Support**: Switch between Gemini and OpenAI-compatible APIs via configuration
- **Model Fallback**: Automatically tries backup models if the primary fails
- **Conversation Memory**: Maintains conversation history per user/group (configurable limit)
- **Keyword Activation**: Only responds to messages starting with configured keywords
- **Taiwan-Localized**: Responds in casual Traditional Chinese (Taiwan)

<img width="750" height="1176" alt="Screenshot 2026-01-24 221339" src="https://github.com/user-attachments/assets/58cfb746-feee-4002-8bd6-1817a958466b" />

## Project Structure

```
turkey_v2/
├── turkey/                 # Main package
│   ├── __init__.py
│   ├── config.py          # Configuration loading
│   ├── llm.py             # LLM client implementations
│   ├── handlers.py        # LINE event handlers
│   └── main.py            # FastAPI application
├── config.yaml            # Application settings
├── .env                   # Secrets (not in git)
├── .env.example           # Example environment file
├── requirements.txt       # Python dependencies
├── run.py                 # Entry point
└── README.md
```

## Setup

### 1. Clone and Install

```bash
git clone https://github.com/fucheng-wang-tw/Turkey.git
cd Turkey

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy example env file
cp .env.example .env

# Edit with your credentials
vim .env
```

Required environment variables:

- `LINE_CHANNEL_ACCESS_TOKEN` - From LINE Developers Console
- `LINE_CHANNEL_SECRET` - From LINE Developers Console
- `API_FORMAT` - Either `gemini` or `openai`
- `GEMINI_API_KEY` - Required if using Gemini
- `OPENAI_API_KEY` - Required if using OpenAI format
- `OPENAI_BASE_URL` - Optional, for OpenAI-compatible endpoints

### 3. Configure Application

Edit `config.yaml` to customize:

```yaml
# Conversation settings
max_history_length: 20

# Keywords that trigger the bot
keywords:
  - "欸"
  - "火雞"

# System prompt
system_prompt: |
  Your custom prompt here...

# LLM settings
temperature: 0.7

# Model lists (tried in order on failure)
gemini_models:
  - "gemini-2.5-flash-lite"
  - "gemini-2.0-flash"

openai_models:
  - "gpt-4o-mini"
  - "gpt-4o"

# Logging
log_file: "/var/log/turkey.log"
```

### 4. Run

```bash
# Development
python run.py

# Production (with uvicorn directly)
uvicorn turkey.main:app --host 0.0.0.0 --port 7777
```

## API Formats

### Gemini (Google GenAI)

```bash
API_FORMAT=gemini
GEMINI_API_KEY=your-api-key
```

### OpenAI

```bash
API_FORMAT=openai
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
```

### OpenAI-Compatible (Local LLMs, Azure, etc.)

```bash
API_FORMAT=openai
OPENAI_API_KEY=any-value
OPENAI_BASE_URL=http://localhost:11434/v1  # Ollama example
```

## Logging

Logs are written to:
- File path configured in `config.yaml` (default: `/var/log/turkey.log`)
- `stderr` (console)

## LINE Webhook

Set your LINE webhook URL to:
```
https://your-domain.com/callback
```

The bot only responds to text messages starting with configured keywords (default: `欸`, `火雞`).

## License

MIT

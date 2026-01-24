"""FastAPI application and routes."""

import sys
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, BackgroundTasks, HTTPException, Header
from linebot.v3 import WebhookParser
from linebot.v3.messaging import AsyncApiClient, AsyncMessagingApi, Configuration
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.webhooks import MessageEvent, TextMessageContent

from .config import settings
from .handlers import handle_message_event

# Configure logging
logging.basicConfig(
    level=settings.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(settings.LOG_FILE),
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger(__name__)

# Validate settings on import
settings.validate()

# LINE Bot SDK configuration
configuration = Configuration(access_token=settings.LINE_CHANNEL_ACCESS_TOKEN)
parser = WebhookParser(settings.LINE_CHANNEL_SECRET)

# Global LINE API client (initialized in lifespan)
line_bot_api: AsyncMessagingApi = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global line_bot_api

    # Startup
    async_api_client = AsyncApiClient(configuration)
    line_bot_api = AsyncMessagingApi(async_api_client)
    logger.info(f"Turkey bot started with {settings.API_FORMAT.upper()} API format")

    yield

    # Shutdown
    logger.info("Turkey bot shutting down...")


app = FastAPI(
    title="Turkey Bot",
    description="LINE chatbot powered by LLM APIs",
    version="2.0.0",
    lifespan=lifespan
)


@app.post("/callback")
async def handle_callback(
    request: Request,
    background_tasks: BackgroundTasks,
    x_line_signature: str = Header(None)
):
    """
    LINE webhook endpoint.
    Verifies signature, returns 200 OK immediately,
    and processes messages in background tasks.
    """
    if x_line_signature is None:
        raise HTTPException(status_code=400, detail="X-Line-Signature header is missing")

    body = await request.body()
    body_str = body.decode('utf-8')

    try:
        events = parser.parse(body_str, x_line_signature)
    except InvalidSignatureError:
        logger.error("Invalid signature. Check your channel secret.")
        raise HTTPException(status_code=400, detail="Invalid signature")

    for event in events:
        if isinstance(event, MessageEvent) and isinstance(event.message, TextMessageContent):
            background_tasks.add_task(handle_message_event, event, line_bot_api)

    return "OK"


@app.get("/")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "running",
        "bot": "Turkey",
        "api_format": settings.API_FORMAT
    }

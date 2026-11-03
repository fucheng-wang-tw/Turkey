"""LLM client implementations for Gemini and OpenAI APIs."""

import logging
from datetime import datetime
from typing import Deque

from google import genai
from google.genai import types
from openai import OpenAI

from .config import settings
from .tools import (
    calculate,
    get_exchange_rate,
    make_web_search,
    random_pick,
    roll_dice,
)

logger = logging.getLogger(__name__)

# Max real web searches per message (extra calls get a 'limit reached' reply)
WEB_SEARCH_MAX_CALLS = 1

WEEKDAYS_ZH = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]


def build_system_prompt() -> str:
    """System prompt with the host's current time appended.

    The time goes at the end so the static prefix stays cacheable.
    """
    now = datetime.now().astimezone()
    return (
        f"{settings.SYSTEM_PROMPT.rstrip()}\n\n"
        f"現在時間：{now.strftime('%Y-%m-%d %H:%M')} {WEEKDAYS_ZH[now.weekday()]}（{now.tzname()}, UTC{now.strftime('%z')}）"
    )

# Initialize clients based on API format
gemini_client = None
openai_client = None

if settings.API_FORMAT == "gemini":
    gemini_client = genai.Client(api_key=settings.GEMINI_API_KEY)
    logger.info(f"Initialized Gemini client with models: {settings.GEMINI_MODELS}")
elif settings.API_FORMAT == "openai":
    openai_client = OpenAI(api_key=settings.OPENAI_API_KEY, base_url=settings.OPENAI_BASE_URL)
    logger.info(f"Initialized OpenAI client at {settings.OPENAI_BASE_URL} with models: {settings.OPENAI_MODELS}")


async def generate_response_gemini(history_queue: Deque[dict], user_text: str) -> str:
    """
    Call Gemini API with history and system prompt.
    Tries models in order until one succeeds.
    """
    contents = []

    # Add historical messages
    for entry in history_queue:
        contents.append(
            types.Content(
                role=entry["role"],
                parts=[types.Part.from_text(text=entry["text"])]
            )
        )

    # Add the current user message
    contents.append(
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_text)]
        )
    )

    system_prompt = build_system_prompt()

    last_error = None
    for model in settings.GEMINI_MODELS:
        try:
            logger.info(f"Trying Gemini model: {model}")
            response = gemini_client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=settings.TEMPERATURE,
                    thinking_config=types.ThinkingConfig(
                        thinking_level=types.ThinkingLevel(settings.GEMINI_THINKING_LEVEL)
                    ),
                    tools=[
                        types.Tool(url_context=types.UrlContext()),
                        random_pick,
                        roll_dice,
                        calculate,
                        get_exchange_rate,
                        *([make_web_search(WEB_SEARCH_MAX_CALLS)] if settings.TAVILY_API_KEY else []),
                    ],
                    # Required to mix built-in tools (url_context) with function calling
                    tool_config=types.ToolConfig(
                        include_server_side_tool_invocations=True
                    ),
                )
            )

            logger.info(f"Gemini response text: {response.text if response else 'None'}")

            if response is None or not hasattr(response, 'text') or response.text is None:
                logger.warning(f"Gemini model {model} returned invalid response")
                continue

            reply_text = response.text.strip()
            if not reply_text:
                logger.warning(f"Gemini model {model} returned empty text")
                continue

            return reply_text

        except Exception as e:
            logger.warning(f"Gemini model {model} failed: {e}")
            last_error = e
            continue

    logger.error(f"All Gemini models failed. Last error: {last_error}", exc_info=True)
    raise Exception(f"All Gemini models failed: {last_error}")


async def generate_response_openai(history_queue: Deque[dict], user_text: str) -> str:
    """
    Call OpenAI-compatible API with history and system prompt.
    Tries models in order until one succeeds.
    """
    # Build messages in OpenAI format
    messages = [{"role": "system", "content": settings.SYSTEM_PROMPT}]

    # Add historical messages (convert 'model' role to 'assistant' for OpenAI)
    for entry in history_queue:
        role = "assistant" if entry["role"] == "model" else entry["role"]
        messages.append({"role": role, "content": entry["text"]})

    # Add current user message
    messages.append({"role": "user", "content": user_text})

    last_error = None
    for model in settings.OPENAI_MODELS:
        try:
            logger.info(f"Trying OpenAI model: {model}")
            response = openai_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=settings.TEMPERATURE
            )

            reply_text = response.choices[0].message.content
            logger.info(f"OpenAI response text: {reply_text}")

            if not reply_text:
                logger.warning(f"OpenAI model {model} returned empty text")
                continue

            return reply_text.strip()

        except Exception as e:
            logger.warning(f"OpenAI model {model} failed: {e}")
            last_error = e
            continue

    logger.error(f"All OpenAI models failed. Last error: {last_error}", exc_info=True)
    raise Exception(f"All OpenAI models failed: {last_error}")


async def generate_response(history_queue: Deque[dict], user_text: str) -> str:
    """Generate a response using the configured API format."""
    if settings.API_FORMAT == "gemini":
        return await generate_response_gemini(history_queue, user_text)
    else:
        return await generate_response_openai(history_queue, user_text)

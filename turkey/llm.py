"""LLM client implementations for Gemini and OpenAI APIs."""

import logging
from typing import Deque

from google import genai
from google.genai import types
from openai import OpenAI

from .config import settings

logger = logging.getLogger(__name__)

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

    last_error = None
    for model in settings.GEMINI_MODELS:
        try:
            logger.info(f"Trying Gemini model: {model}")
            response = gemini_client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=settings.SYSTEM_PROMPT,
                    temperature=settings.TEMPERATURE,
                    thinking_config=types.ThinkingConfig(
                        thinking_level=types.ThinkingLevel.HIGH
                    ),
                    tools=[
                        types.Tool(url_context=types.UrlContext()),
                        types.Tool(google_search=types.GoogleSearch()),
                    ],
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

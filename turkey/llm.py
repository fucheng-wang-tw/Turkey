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

# Max Gemini requests per model attempt. The last request (or any request after
# a web search) disables function calling, so the model must answer in text.
MAX_GEMINI_REQUESTS = 3

# Max real web searches per message (guards against parallel web_search calls)
WEB_SEARCH_MAX_CALLS = 1

TOOL_LIMIT_MESSAGE = "Tool limit reached for this message. Do not call any tools. Answer the user now."
ANSWER_NOW_NUDGE = "請根據以上工具結果直接回答使用者，不要再呼叫任何工具。資料不完整的話，就說明哪個部分不確定。"

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


def _call_tool(function_call: types.FunctionCall, functions: dict) -> dict:
    """Run one function call from the model and return its result."""
    func = functions.get(function_call.name)
    if func is None:
        return {"error": f"Unknown tool: {function_call.name}"}
    try:
        return func(**(function_call.args or {}))
    except Exception as e:
        logger.warning(f"Tool {function_call.name} failed: {e}")
        return {"error": f"Tool failed: {e}"}


def _generate_with_tools(model: str, contents: list, system_prompt: str) -> types.GenerateContentResponse:
    """
    Call Gemini with a manual function-calling loop.

    After a web search, the next request uses function-calling mode NONE so the
    model answers right away instead of searching again.
    """
    functions = [random_pick, roll_dice, calculate, get_exchange_rate]
    if settings.TAVILY_API_KEY:
        functions.append(make_web_search(WEB_SEARCH_MAX_CALLS))
    function_map = {f.__name__: f for f in functions}

    contents = list(contents)
    searched = False
    for request_no in range(1, MAX_GEMINI_REQUESTS + 1):
        tools_allowed = not searched and request_no < MAX_GEMINI_REQUESTS
        mode = types.FunctionCallingConfigMode.AUTO if tools_allowed else types.FunctionCallingConfigMode.NONE

        response = gemini_client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=settings.TEMPERATURE,
                thinking_config=types.ThinkingConfig(
                    thinking_level=types.ThinkingLevel(settings.GEMINI_THINKING_LEVEL)
                ),
                tools=[types.Tool(url_context=types.UrlContext()), *functions],
                tool_config=types.ToolConfig(
                    # Required to mix built-in tools (url_context) with function calling
                    include_server_side_tool_invocations=True,
                    function_calling_config=types.FunctionCallingConfig(mode=mode),
                ),
                # We run the tools ourselves (see loop below)
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            ),
        )

        function_calls = response.function_calls
        if not function_calls:
            return response

        if tools_allowed:
            logger.info(f"Gemini request {request_no}: tool calls {[fc.name for fc in function_calls]}")
        else:
            # The model sometimes ignores mode=NONE. Instead of failing, answer
            # its calls with "limit reached" and ask once more.
            logger.warning(
                f"Gemini request {request_no}: model ignored mode=NONE and called "
                f"{[fc.name for fc in function_calls]}"
            )
            if request_no == MAX_GEMINI_REQUESTS:
                return response

        # Keep the model turn as-is (it carries thought signatures and
        # server-side tool records), then add our tool results.
        contents.append(response.candidates[0].content)
        result_parts = []
        for fc in function_calls:
            if not tools_allowed:
                result = {"error": TOOL_LIMIT_MESSAGE}
            else:
                if fc.name == "web_search":
                    searched = True
                result = _call_tool(fc, function_map)
            result_parts.append(
                types.Part(
                    function_response=types.FunctionResponse(
                        id=fc.id,
                        name=fc.name,
                        response=result,
                    )
                )
            )

        # If the next request will disable tools, say so explicitly next to
        # the results; the model follows this better than mode=NONE alone.
        next_tools_allowed = not searched and request_no + 1 < MAX_GEMINI_REQUESTS
        if not next_tools_allowed:
            result_parts.append(types.Part.from_text(text=ANSWER_NOW_NUDGE))

        contents.append(types.Content(role="user", parts=result_parts))

    return response


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
            response = _generate_with_tools(model, contents, system_prompt)

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

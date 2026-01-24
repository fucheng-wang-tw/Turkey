"""LINE message event handlers."""

import logging
from collections import deque, defaultdict
from typing import Dict, Deque

from linebot.v3.messaging import (
    AsyncMessagingApi,
    ReplyMessageRequest,
    TextMessage,
    ShowLoadingAnimationRequest
)
from linebot.v3.webhooks import MessageEvent

from .config import settings
from .llm import generate_response

logger = logging.getLogger(__name__)

# Conversation History Storage
HISTORY_STORE: Dict[str, Deque[dict]] = defaultdict(
    lambda: deque(maxlen=settings.MAX_HISTORY_LENGTH)
)


def get_conversation_id(event: MessageEvent) -> str:
    """
    Determine the unique ID for the conversation.
    Uses group/room ID if available, otherwise user ID.
    """
    if event.source.type == "group":
        return event.source.group_id
    elif event.source.type == "room":
        return event.source.room_id
    else:
        return event.source.user_id


async def handle_message_event(event: MessageEvent, line_bot_api: AsyncMessagingApi) -> None:
    """
    Process a LINE message event.
    Filters by keyword, generates LLM response, and replies.
    """
    try:
        user_text = event.message.text
        conversation_id = get_conversation_id(event)

        # Check if message starts with any keyword
        if not any(user_text.startswith(keyword) for keyword in settings.KEYWORDS):
            logger.info(f"Message from {conversation_id} does not start with keywords: {user_text}")
            return

        # Remove the keyword prefix from the message
        for keyword in settings.KEYWORDS:
            if user_text.startswith(keyword):
                user_text = user_text[len(keyword):].strip()
                break

        # Show loading animation
        try:
            await line_bot_api.show_loading_animation(
                ShowLoadingAnimationRequest(chat_id=conversation_id, loading_seconds=5)
            )
        except Exception as e:
            logger.warning(f"Could not show loading animation: {e}")

        # Get conversation history and generate response
        history_queue = HISTORY_STORE[conversation_id]

        try:
            reply_text = await generate_response(history_queue, user_text)

            # Update history with successful response
            history_queue.append({"role": "user", "text": user_text})
            history_queue.append({"role": "model", "text": reply_text})

            logger.info(f"Generated reply: {reply_text}")

        except Exception as e:
            logger.error(f"LLM Error: {e}", exc_info=True)
            reply_text = "Sorry, I'm having trouble thinking right now. Please try again later."

        # Reply to LINE
        await line_bot_api.reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=reply_text)]
            )
        )

    except Exception as e:
        logger.error(f"Error handling event: {e}", exc_info=True)

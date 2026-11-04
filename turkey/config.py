"""Configuration management for Turkey bot."""

import os
import sys
import logging
from pathlib import Path
from typing import List

import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Determine project root (parent of turkey package)
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

# Load application config from YAML
if not CONFIG_PATH.exists():
    print(f"Error: config.yaml not found at {CONFIG_PATH}")
    sys.exit(1)

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    _config = yaml.safe_load(f)


class Settings:
    """Application settings loaded from environment and config file."""

    # LINE credentials (from .env)
    LINE_CHANNEL_ACCESS_TOKEN: str = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
    LINE_CHANNEL_SECRET: str = os.getenv("LINE_CHANNEL_SECRET", "")

    # API format: "gemini" or "openai"
    API_FORMAT: str = os.getenv("API_FORMAT", "gemini").lower()

    # Gemini configuration
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODELS: List[str] = _config.get("gemini_models", ["gemini-2.5-flash-lite"])
    GEMINI_THINKING_LEVEL: str = str(_config.get("gemini_thinking_level", "low")).upper()

    # OpenAI configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL: str = os.getenv(
        "OPENAI_BASE_URL",
        _config.get("openai_base_url", "https://api.openai.com/v1")
    )
    OPENAI_MODELS: List[str] = _config.get("openai_models", ["gpt-4o-mini"])

    # Tavily web search (optional; enables the web_search tool for Gemini)
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")

    # Application settings (from config.yaml)
    SYSTEM_PROMPT: str = _config.get("system_prompt", "You are a helpful assistant.")
    MAX_HISTORY_LENGTH: int = _config.get("max_history_length", 20)
    KEYWORDS: List[str] = _config.get("keywords", ["欸", "火雞"])
    ERROR_MESSAGE: str = _config.get(
        "error_message",
        "Sorry, I'm having trouble thinking right now. Please try again later."
    )
    TEMPERATURE: float = _config.get("temperature", 0.7)

    # Logging
    LOG_LEVEL: int = logging.INFO
    LOG_FILE: str = _config.get("log_file", "/var/log/turkey.log")

    @classmethod
    def validate(cls) -> None:
        """Validate required settings."""
        if not cls.LINE_CHANNEL_ACCESS_TOKEN or not cls.LINE_CHANNEL_SECRET:
            logging.error("LINE_CHANNEL_ACCESS_TOKEN and LINE_CHANNEL_SECRET must be set.")
            sys.exit(1)

        if cls.API_FORMAT == "gemini" and not cls.GEMINI_API_KEY:
            logging.error("GEMINI_API_KEY must be set when using Gemini format.")
            sys.exit(1)

        if cls.API_FORMAT == "openai" and not cls.OPENAI_API_KEY:
            logging.error("OPENAI_API_KEY must be set when using OpenAI format.")
            sys.exit(1)

        if cls.GEMINI_THINKING_LEVEL not in ("MINIMAL", "LOW", "MEDIUM", "HIGH"):
            logging.error(
                f"Unknown gemini_thinking_level '{cls.GEMINI_THINKING_LEVEL.lower()}'. "
                "Must be one of: minimal, low, medium, high."
            )
            sys.exit(1)

        if cls.API_FORMAT not in ("gemini", "openai"):
            logging.error(f"Unknown API_FORMAT '{cls.API_FORMAT}'. Must be 'gemini' or 'openai'.")
            sys.exit(1)


settings = Settings()

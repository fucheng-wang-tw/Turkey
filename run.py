#!/usr/bin/env python3
"""Entry point for Turkey bot."""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "turkey.main:app",
        host="0.0.0.0",
        port=7777,
        reload=False
    )

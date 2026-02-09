"""Gemini API client with retry logic."""

import os
import time
import random
from typing import Optional
from loguru import logger

from google import genai
from google.genai import types

from ...config.constants import GEMINI


class GeminiClient:
    """Gemini API client with retry and thinking mode support."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not set")
        self.client = genai.Client(api_key=self.api_key)

    def generate(
        self,
        prompt: str,
        model: str = None,
        thinking: Optional[str] = None,
        max_retries: int = None,
    ) -> Optional[str]:
        """
        Generate content with retry logic.

        Args:
            prompt: The prompt to send
            model: Model ID (default: gemini-3-flash-preview)
            thinking: Thinking level (LOW/MEDIUM/HIGH) or None
            max_retries: Number of retries
        """
        model = model or GEMINI["MODEL_FLASH_3"]
        max_retries = max_retries or GEMINI["MAX_RETRIES"]
        thinking_budget = None

        for attempt in range(max_retries):
            try:
                config_params = {"temperature": 1.0}

                if thinking and "preview" in model.lower():
                    thinking_budget = GEMINI["THINKING_BUDGET"].get(thinking.upper(), 1024)
                    config_params["thinking_config"] = types.ThinkingConfig(
                        thinking_budget=thinking_budget
                    )

                start_ts = time.time()
                response = self.client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(**config_params),
                )
                latency_ms = int((time.time() - start_ts) * 1000)

                if response and response.text:
                    logger.info(
                        f"GEMINI_REQUEST: model={model} prompt_chars={len(prompt)} "
                        f"thinking_budget={thinking_budget} latency_ms={latency_ms}"
                    )
                    return response.text.strip()

                logger.warning(
                    f"GEMINI_EMPTY: model={model} prompt_chars={len(prompt)} latency_ms={latency_ms}"
                )

            except Exception as e:
                err_str = str(e).lower()
                error_type = "unknown"

                if "503" in err_str or "overloaded" in err_str:
                    wait = (2 ** attempt) + random.uniform(0, 1)
                    error_type = "overloaded"
                elif "429" in err_str or "rate" in err_str:
                    wait = (2 ** attempt) * 5 + random.uniform(0, 5)
                    error_type = "rate_limit"
                elif "timeout" in err_str:
                    wait = 10
                    error_type = "timeout"
                else:
                    wait = (2 ** attempt) + random.uniform(0, 1)

                logger.warning(
                    f"GEMINI_RETRY: attempt={attempt+1}/{max_retries} error={error_type} "
                    f"wait_ms={int(wait*1000)} model={model}"
                )
                time.sleep(wait)

        logger.error(f"GEMINI_FAILED: model={model} max_retries={max_retries} exhausted")
        return None

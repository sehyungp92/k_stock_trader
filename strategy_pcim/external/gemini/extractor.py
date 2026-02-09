"""Gemini-based signal extraction."""

import json
from typing import List, Optional
from dataclasses import dataclass
from loguru import logger

from .client import GeminiClient
from .prompts import SIGNAL_EXTRACTION_PROMPT, PUNCTUATION_PROMPT
from ...config.constants import SIGNAL_EXTRACTION


@dataclass
class ExtractedSignal:
    """Extracted trading signal from video."""
    company_name: str
    ticker: Optional[str]  # None if LLM couldn't determine
    conviction_score: float  # 0.0 to 1.0


@dataclass
class ExtractionResult:
    """Result of signal extraction."""
    video_summary: str
    signals: List[ExtractedSignal]
    raw_response: str


class SignalExtractor:
    """Extracts trading signals from video transcripts using Gemini."""

    def __init__(self, gemini_client: GeminiClient):
        self.client = gemini_client

    def punctuate_transcript(self, raw_transcript: str) -> str:
        """Clean up raw transcript with proper punctuation and formatting."""
        try:
            prompt = PUNCTUATION_PROMPT.format(transcript=raw_transcript)
            result = self.client.generate(
                prompt,
                model="gemini-2.5-flash",
                thinking=None,
            )
            if result:
                logger.debug(f"PUNCTUATION: input={len(raw_transcript)} output={len(result)} chars")
                return result
            return raw_transcript
        except Exception as e:
            logger.warning(f"Punctuation failed: {e}")
            return raw_transcript

    def extract_signals(
        self,
        transcript: str,
        video_id: str = None,
        conviction_threshold: float = None
    ) -> Optional[ExtractionResult]:
        """Extract trading signals from transcript.

        Args:
            transcript: The video transcript text
            video_id: Optional video ID for logging
            conviction_threshold: Minimum conviction score to accept (default from config)

        Returns:
            ExtractionResult with signals that meet threshold, or None on failure
        """
        if conviction_threshold is None:
            conviction_threshold = SIGNAL_EXTRACTION["CONVICTION_THRESHOLD"]

        prompt = SIGNAL_EXTRACTION_PROMPT.format(transcript=transcript)

        try:
            response = self.client.generate(
                prompt,
                model="gemini-3-flash-preview",
                thinking="MEDIUM",
            )

            if not response:
                logger.error(f"SIGNAL_EXTRACTION: video_id={video_id} empty_response=True")
                return None

            json_str = response
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0]

            data = json.loads(json_str.strip())

            signals = []
            rejected = []
            for rec in data.get("recommendations", []):
                company_name = rec.get("company_name", "").strip()
                ticker = rec.get("ticker")
                # Handle ticker - could be string or null
                if ticker is not None:
                    ticker = str(ticker).strip()
                    if not ticker or ticker.lower() == "null":
                        ticker = None

                conviction_score = rec.get("conviction_score", 0.0)
                # Ensure float type
                try:
                    conviction_score = float(conviction_score)
                except (ValueError, TypeError):
                    conviction_score = 0.0

                if not company_name:
                    continue

                if conviction_score >= conviction_threshold:
                    signals.append(ExtractedSignal(
                        company_name=company_name,
                        ticker=ticker,
                        conviction_score=conviction_score,
                    ))
                else:
                    rejected.append({
                        "company_name": company_name,
                        "ticker": ticker,
                        "conviction_score": conviction_score,
                        "reason": f"below_threshold_{conviction_threshold}"
                    })

            above_09 = sum(1 for s in signals if s.conviction_score >= 0.9)
            above_07 = sum(1 for s in signals if 0.7 <= s.conviction_score < 0.9)
            with_ticker = sum(1 for s in signals if s.ticker)
            logger.info(
                f"SIGNAL_EXTRACTION: video_id={video_id} signals={len(signals)} "
                f"(>=0.9:{above_09} >=0.7:{above_07}) with_ticker={with_ticker} rejected={len(rejected)}"
            )

            return ExtractionResult(
                video_summary=data.get("video_summary", ""),
                signals=signals,
                raw_response=response,
            )

        except json.JSONDecodeError as e:
            logger.error(f"SIGNAL_EXTRACTION: video_id={video_id} json_parse_error={e}")
            logger.debug(f"Raw response: {response[:500]}")
            return None
        except Exception as e:
            logger.error(f"SIGNAL_EXTRACTION: video_id={video_id} error={e}")
            return None

"""Transcript extraction using yt-dlp."""

import os
import json
from typing import Optional
from loguru import logger


def extract_video_id(url: str) -> str:
    if "youtube.com/watch?v=" in url:
        return url.split("watch?v=")[1].split("&")[0]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    return url


def fetch_transcript_ytdlp(video_url: str, use_cookies: bool = False, browser: str = 'chrome') -> Optional[str]:
    """Extract transcript using yt-dlp subtitle extraction.

    Returns:
        - Transcript string on success
        - "COOKIES_NEEDED" sentinel if bot detection requires cookies
        - None on failure
    """
    try:
        import yt_dlp
    except ImportError:
        logger.error("yt-dlp not installed")
        return None

    video_id = extract_video_id(video_url)

    ydl_opts = {
        'writeautomaticsub': True,
        'writesubtitles': True,
        'skip_download': True,
        'subtitleslangs': ['ko', 'en', 'en-US', 'en-GB'],
        'subtitlesformat': 'json3',
        'outtmpl': f'temp_subtitle_{video_id}',
        'quiet': True,
        'no_warnings': True,
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
        'extractor_args': {'youtube': {'player_client': ['android', 'web']}},
        'ignoreerrors': True,
    }

    if use_cookies:
        ydl_opts['cookiesfrombrowser'] = (browser,)

    extraction_error = None
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.extract_info(video_url, download=True)
    except Exception as e:
        extraction_error = e
        error_str = str(e).lower()
        # Check for bot detection / sign-in requirement
        if 'bot' in error_str or 'sign in' in error_str or 'confirm' in error_str:
            if not use_cookies:
                logger.debug(f"Bot detection for {video_id}, will signal for cookie retry")

    # Check for subtitle files even if extraction raised an exception
    # (partial downloads can still produce usable files)
    subtitle_files = [f for f in os.listdir('.')
                      if f.startswith(f'temp_subtitle_{video_id}') and f.endswith('.json3')]

    if not subtitle_files:
        # No files produced - check if we should signal for cookie retry
        if extraction_error:
            error_str = str(extraction_error).lower()
            if ('bot' in error_str or 'sign in' in error_str or 'confirm' in error_str) and not use_cookies:
                logger.info(f"TRANSCRIPT: video_id={video_id} status=COOKIES_NEEDED")
                return "COOKIES_NEEDED"
            logger.debug(f"yt-dlp extraction error: {extraction_error}")
        logger.debug(f"TRANSCRIPT: video_id={video_id} status=NO_SUBTITLES")
        return None

    # Prefer Korean subtitles, fallback to any available
    korean_files = [f for f in subtitle_files if '.ko.' in f]
    english_files = [f for f in subtitle_files if '.en' in f]

    if korean_files:
        selected_file = korean_files[0]
        lang = "ko"
    elif english_files:
        selected_file = english_files[0]
        lang = "en"
    else:
        selected_file = subtitle_files[0]
        lang = "unknown"

    try:
        with open(selected_file, 'r', encoding='utf-8') as f:
            subtitle_data = json.load(f)

        transcript = []
        for event in subtitle_data.get('events', []):
            if 'segs' in event:
                text = ''.join(seg.get('utf8', '') for seg in event['segs'])
                if text.strip():
                    transcript.append(text.strip())

        result = ' '.join(transcript)
        logger.info(f"TRANSCRIPT: video_id={video_id} lang={lang} chars={len(result)}")
        return result
    except json.JSONDecodeError as e:
        logger.warning(f"TRANSCRIPT: video_id={video_id} status=JSON_PARSE_ERROR error={e}")
        return None
    finally:
        for f in subtitle_files:
            try:
                os.remove(f)
            except Exception:
                pass


def fetch_transcript(video_url: str) -> Optional[str]:
    """Fetch transcript from YouTube video.

    Handles cookie retry automatically when bot detection is encountered.
    """
    transcript = fetch_transcript_ytdlp(video_url)

    # Handle cookie retry sentinel
    if transcript == "COOKIES_NEEDED":
        logger.info(f"Retrying with cookies: {video_url}")
        transcript = fetch_transcript_ytdlp(video_url, use_cookies=True, browser='chrome')
        if transcript == "COOKIES_NEEDED":
            # Still failing, try different browser
            transcript = fetch_transcript_ytdlp(video_url, use_cookies=True, browser='firefox')

    if transcript and transcript != "COOKIES_NEEDED":
        return transcript

    logger.warning(f"No transcript found for {video_url}")
    return None

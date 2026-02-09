"""YouTube channel monitor using RSS + yt-dlp."""

import copy
import feedparser
import json
import os
import urllib.request
from collections import defaultdict
from datetime import datetime, time
from typing import Dict, List, Set
from loguru import logger
import pytz

from .models import ChannelConfig, VideoInfo
from ...config.constants import YOUTUBE
from kis_core.trading_calendar import get_trading_calendar


class YouTubeWatcher:
    """Monitors YouTube channels for new videos via RSS + yt-dlp fallback."""

    def __init__(self, channels: List[ChannelConfig], state_file: str = None):
        self.channels = channels
        self.state_file = state_file or YOUTUBE["STATE_FILE"]
        self.channel_states = self._load_state()
        self.kst = pytz.timezone("Asia/Seoul")

    def _load_state(self) -> dict:
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading state: {e}")
        return {}

    def _save_state(self) -> None:
        try:
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump({
                    **self.channel_states,
                    'last_save': datetime.now().isoformat()
                }, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving state: {e}")

    def _fetch_from_rss(self, channel_id: str, max_videos: int) -> List[VideoInfo]:
        rss_url = f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
        try:
            req = urllib.request.Request(
                rss_url,
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                feed_data = response.read()

            feed = feedparser.parse(feed_data)
            if not feed.entries:
                return []

            videos = []
            for entry in feed.entries[:max_videos]:
                videos.append(VideoInfo(
                    video_id=entry.yt_videoid,
                    channel_id=channel_id,
                    channel_name=entry.author,
                    title=entry.title,
                    url=entry.link,
                    published=entry.published,
                    description=getattr(entry, 'summary', ''),
                    source='rss',
                ))
            return videos
        except Exception as e:
            logger.debug(f"RSS error for {channel_id}: {e}")
            return []

    def _fetch_from_ytdlp(self, channel_id: str, max_videos: int) -> List[VideoInfo]:
        try:
            import yt_dlp
        except ImportError:
            logger.warning("yt-dlp not installed")
            return []

        videos = []
        tabs = [
            f"https://www.youtube.com/channel/{channel_id}/videos",
            f"https://www.youtube.com/channel/{channel_id}/streams",
        ]

        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,
            'playlistend': max_videos,
            'ignoreerrors': True,
            'socket_timeout': 30,
        }

        for tab_url in tabs:
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    result = ydl.extract_info(tab_url, download=False)
                    if result and 'entries' in result:
                        for entry in result['entries']:
                            if not entry or not entry.get('id'):
                                continue
                            video_id = entry['id']
                            upload_date = entry.get('upload_date', '')
                            if upload_date:
                                try:
                                    parsed = datetime.strptime(upload_date, '%Y%m%d')
                                    published = parsed.strftime('%Y-%m-%dT23:59:59+00:00')
                                except Exception:
                                    published = ''
                            else:
                                published = ''
                            videos.append(VideoInfo(
                                video_id=video_id,
                                channel_id=channel_id,
                                channel_name=entry.get('uploader', ''),
                                title=entry.get('title', 'Unknown'),
                                url=f"https://www.youtube.com/watch?v={video_id}",
                                published=published,
                                source='ytdlp',
                                is_live=entry.get('live_status') == 'was_live',
                            ))
            except Exception as e:
                logger.debug(f"yt-dlp error for {tab_url}: {e}")

        return videos

    def _is_in_valid_window(self, published_str: str) -> bool:
        """Check if video is in valid signal window.

        Valid window: previous trading day 15:00 KST → current trading day 08:30 KST

        Examples:
        - Normal Tuesday: Mon 15:00 → Tue 08:30
        - Monday: Fri 15:00 → Mon 08:30 (includes weekend)
        - After holiday: Fri 15:00 → Tue 08:30
        """
        if not published_str:
            return True  # If no publish time, accept (conservative)

        try:
            published = datetime.fromisoformat(published_str.replace('Z', '+00:00'))
            published_kst = published.astimezone(self.kst)
        except Exception:
            return True  # Parse error, accept (conservative)

        now_kst = datetime.now(self.kst)
        today = now_kst.date()
        calendar = get_trading_calendar()

        morning_cutoff = time(
            YOUTUBE["VIDEO_MORNING_CUTOFF_HOUR"],
            YOUTUBE["VIDEO_MORNING_CUTOFF_MIN"]
        )

        # Determine which trading day we're targeting
        if now_kst.time() <= morning_cutoff and calendar.is_trading_day(today):
            # Before 08:30 on a trading day - we're still in the window for today
            current_trading_day = today
        else:
            # After 08:30 or not a trading day - target next trading day
            if calendar.is_trading_day(today):
                current_trading_day = calendar.next_trading_day(today)
            else:
                current_trading_day = calendar.next_trading_day(today)

        prev_trading_day = calendar.previous_trading_day(current_trading_day)

        # Build window boundaries
        window_start = datetime(
            prev_trading_day.year,
            prev_trading_day.month,
            prev_trading_day.day,
            YOUTUBE["VIDEO_CUTOFF_HOUR"],
            0, 0,
            tzinfo=self.kst
        )
        window_end = datetime(
            current_trading_day.year,
            current_trading_day.month,
            current_trading_day.day,
            YOUTUBE["VIDEO_MORNING_CUTOFF_HOUR"],
            YOUTUBE["VIDEO_MORNING_CUTOFF_MIN"],
            0,
            tzinfo=self.kst
        )

        return window_start <= published_kst <= window_end

    def fetch_videos(self, channel: ChannelConfig) -> List[VideoInfo]:
        max_videos = YOUTUBE["MAX_VIDEOS_PER_CHANNEL"]
        seen_ids: Set[str] = set()
        videos = []

        rss_videos = self._fetch_from_rss(channel.channel_id, max_videos)
        rss_ok = len(rss_videos) > 0
        for video in rss_videos:
            if video.video_id not in seen_ids:
                videos.append(video)
                seen_ids.add(video.video_id)

        rss_count = len(videos)
        ytdlp_videos = self._fetch_from_ytdlp(channel.channel_id, max_videos)
        used_fallback = len(ytdlp_videos) > 0 and not rss_ok
        for video in ytdlp_videos:
            if video.video_id not in seen_ids:
                videos.append(video)
                seen_ids.add(video.video_id)

        ytdlp_new = len(videos) - rss_count
        logger.info(
            f"YOUTUBE_FETCH: channel={channel.channel_id} name={channel.name} "
            f"total_videos={len(videos)} rss_ok={rss_ok} rss_count={rss_count} "
            f"ytdlp_new={ytdlp_new} ytdlp_fallback={used_fallback}"
        )

        videos.sort(key=lambda x: x.published, reverse=True)
        return videos[:max_videos]

    def _fetch_new_for_channel_id(self, channel_id: str, display_name: str) -> List[VideoInfo]:
        """Fetch new videos for a channel_id, manage seen-state. No keyword filtering."""
        # Use a temporary ChannelConfig just for fetching (notify_all=True to skip filtering)
        fetch_cfg = ChannelConfig(channel_id=channel_id, name=display_name, notify_all=True)
        logger.info(f"Checking channel: {display_name} ({channel_id})")
        videos = self.fetch_videos(fetch_cfg)
        if not videos:
            return []

        last_video_id = self.channel_states.get(channel_id)

        if last_video_id is None:
            logger.info(f"  Initial check for {display_name}")
            self.channel_states[channel_id] = videos[0].video_id
            self._save_state()
            return []

        new_videos = []
        for video in videos:
            if video.video_id == last_video_id:
                break
            if not self._is_in_valid_window(video.published):
                continue
            new_videos.append(video)

        if videos:
            self.channel_states[channel_id] = videos[0].video_id
            self._save_state()

        logger.info(f"  Found {len(new_videos)} new videos")
        return new_videos

    def check_all_channels(self) -> List[VideoInfo]:
        """Fetch new videos and match against influencer configs.

        Same channel_id is fetched only once, then each video is matched
        against every config entry for that channel (keyword filtering).
        This allows the same channel to map to multiple influencers.
        """
        # Group configs by channel_id to fetch once per channel
        by_channel: Dict[str, List[ChannelConfig]] = defaultdict(list)
        for ch in self.channels:
            by_channel[ch.channel_id].append(ch)

        all_videos: List[VideoInfo] = []

        for channel_id, configs in by_channel.items():
            display_name = configs[0].name
            raw_new = self._fetch_new_for_channel_id(channel_id, display_name)

            for video in raw_new:
                for cfg in configs:
                    if cfg.should_process(video.title):
                        matched = copy.copy(video)
                        matched.channel_name = cfg.name
                        matched.influencer_id = cfg.influencer_id
                        all_videos.append(matched)

        return all_videos

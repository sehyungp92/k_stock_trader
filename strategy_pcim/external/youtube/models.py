"""YouTube data models."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class ChannelConfig:
    """Configuration for a monitored channel.

    influencer_id is NOT unique — the same influencer can appear on
    multiple channels, and the same channel can host multiple influencers.
    """
    channel_id: str
    name: str
    influencer_id: str = ""
    keywords: List[str] = field(default_factory=list)
    notify_all: bool = True

    def should_process(self, video_title: str) -> bool:
        if self.notify_all:
            return True
        if not self.keywords:
            return True
        title_lower = video_title.lower()
        return any(kw.lower() in title_lower for kw in self.keywords)


@dataclass
class VideoInfo:
    """Information about a YouTube video."""
    video_id: str
    channel_id: str
    channel_name: str
    title: str
    url: str
    published: str
    description: str = ""
    source: str = "rss"
    is_live: bool = False
    influencer_id: str = ""

    @property
    def link(self) -> str:
        return f"https://www.youtube.com/watch?v={self.video_id}"

"""Shared fixtures for LivePhish provider tests."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from music_assistant.providers.livephish import SUPPORTED_FEATURES, LivePhishProvider
from music_assistant.providers.livephish.client import LivePhishClient
from tests.providers.livephish.fixtures import ARTISTS, RELEASES


@pytest.fixture
def provider() -> LivePhishProvider:
    """Create a real provider with mocked catalog transport."""
    config = MagicMock()
    config.instance_id = "livephish--test"
    config.get_value.side_effect = lambda key, default=None: {
        "log_level": "GLOBAL",
    }.get(key, default)
    manifest = MagicMock()
    manifest.domain = "livephish"
    provider = LivePhishProvider(MagicMock(), manifest, config, SUPPORTED_FEATURES)
    provider._client = MagicMock(spec=LivePhishClient)
    provider._client.artists = AsyncMock(return_value=ARTISTS)
    provider._client.favorite_albums = AsyncMock(return_value=[])
    provider._client.artist_releases = AsyncMock(return_value=RELEASES)
    return provider

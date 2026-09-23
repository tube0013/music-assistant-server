"""Album metadata cache bounds and concurrency protection."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from music_assistant.providers.livephish.client import LivePhishClient, LivePhishError


@pytest.fixture
def client() -> LivePhishClient:
    """Use isolated cache state and synthetic network transport."""
    return LivePhishClient(MagicMock(), "", "")


async def test_concurrent_misses_share_one_request(client: LivePhishClient) -> None:
    """Simultaneous requests for one show cannot duplicate the network lookup."""
    started = asyncio.Event()
    release = asyncio.Event()

    async def fetch(*_args: object, **_kwargs: object) -> dict[str, object]:
        started.set()
        await release.wait()
        return {"Response": {"containerID": 1747, "tracks": []}}

    with patch.object(client, "_request", new_callable=AsyncMock, side_effect=fetch) as request:
        first = asyncio.create_task(client.album("1747"))
        await started.wait()
        second = asyncio.create_task(client.album("1747"))
        release.set()
        a, b = await asyncio.gather(first, second)
        assert a is b
        request.assert_awaited_once()


async def test_cache_evicts_at_bound(client: LivePhishClient) -> None:
    """Inserting more shows than the budget retains only a bounded set."""
    with (
        patch("music_assistant.providers.livephish.client.MAX_CACHED_ALBUMS", 2),
        patch.object(client, "_request", new_callable=AsyncMock) as request,
    ):
        request.side_effect = [{"Response": {"containerID": i, "tracks": []}} for i in (1, 2, 3, 1)]
        for i in (1, 2, 3):
            await client.album(str(i))
        assert list(client._albums) == ["2", "3"]
        await client.album("1")
        assert len(client._albums) == 2
        assert request.await_count == 4


async def test_expired_cache_refreshes(client: LivePhishClient) -> None:
    """Expired entries are replaced and failed responses never enter the cache."""
    client._albums["1"] = (0, {"containerID": 1, "tracks": []})
    with patch.object(client, "_request", new_callable=AsyncMock) as request:
        request.side_effect = [
            LivePhishError("catalog: HTTP 503"),
            {"Response": {"containerID": 1, "tracks": [], "updated": True}},
        ]
        with pytest.raises(LivePhishError):
            await client.album("1")
        assert (await client.album("1"))["updated"] is True
        request.assert_awaited()

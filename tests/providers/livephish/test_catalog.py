"""Catalog protocol regression tests with synthetic service responses."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from music_assistant.providers.livephish.client import LivePhishClient, LivePhishError


@pytest.fixture
def client() -> LivePhishClient:
    """Create a catalog client without real credentials or network access."""
    return LivePhishClient(MagicMock(), "", "")


async def test_artist_cache_coalesces_requests(client: LivePhishClient) -> None:
    """Simultaneous browsing and search share one catalog fetch."""
    artists = [{"artistID": 19, "artistName": "Oysterhead"}]
    with patch.object(client, "_request", new_callable=AsyncMock) as request:
        request.return_value = {"Response": {"artists": artists}}
        first, second = await asyncio.gather(client.artists(), client.artists())
        assert first == artists
        assert second == artists
        request.assert_awaited_once()


async def test_expired_catalog_cache_refreshes(client: LivePhishClient) -> None:
    """Expired metadata is refreshed while fresh entries are reused."""
    with patch.object(client, "_request", new_callable=AsyncMock) as request:
        request.return_value = {"Response": {"artists": []}}
        await client.artists()
        key = next(iter(client._catalog_cache))
        client._catalog_cache[key] = (0, {"Response": {"artists": []}})
        await client.artists()
        assert request.await_count == 2


async def test_release_request_filters_and_offset(client: LivePhishClient) -> None:
    """Artist and year filters retain one-based service paging."""
    with patch.object(client, "_request", new_callable=AsyncMock) as request:
        request.return_value = {"Response": {"containers": [{"containerID": 1773}]}}
        await client.releases("19", "2022", offset=100)
        assert request.call_args.kwargs["params"] == {
            "method": "catalog.containersAll",
            "artistList": "19",
            "showYears": "2022",
            "startOffset": 101,
            "limit": 100,
            "availtype": 1,
            "sortBy": "performanceDate",
            "sortType": "desc",
        }


async def test_paging_ignores_page_local_total(client: LivePhishClient) -> None:
    """A misleading total must not truncate an artist after the first full page."""
    first = [{"containerID": number} for number in range(1, 101)]
    with patch.object(client, "_request", new_callable=AsyncMock) as request:
        request.side_effect = [
            {"Response": {"containers": first, "totalMatchedRecords": 100}},
            {"Response": {"containers": [{"containerID": 101}], "totalMatchedRecords": 1}},
            {"Response": {"containers": [], "totalMatchedRecords": 0}},
        ]
        assert len(await client.artist_releases("1")) == 101
        assert request.await_args_list[1].kwargs["params"]["startOffset"] == 101


async def test_repeated_page_fails_instead_of_looping(client: LivePhishClient) -> None:
    """Fail clearly if the service ignores pagination."""
    page = [{"containerID": number} for number in range(1, 101)]
    with patch.object(client, "releases", new_callable=AsyncMock, return_value=page) as releases:
        with pytest.raises(LivePhishError, match="pagination did not advance"):
            await client.artist_releases("1")
        assert releases.await_count == 2


async def test_search_flattens_sections(client: LivePhishClient) -> None:
    """Preserve show and song matches from separate search groups."""
    items = [{"containerID": 1747}, {"containerID": 1747, "trackID": 29990}]
    with patch.object(client, "_request", new_callable=AsyncMock) as request:
        request.return_value = {
            "Response": {
                "searchError": 0,
                "catalogSearchTypeContainers": [
                    {"catalogSearchContainers": [{"catalogSearchResultItems": [item]}]}
                    for item in items
                ],
            }
        }
        assert await client.search(" Axilla ") == items
        assert request.call_args.kwargs["params"]["searchStr"] == "Axilla"


@pytest.mark.parametrize("response", [None, "error", {}, {"artists": [None]}])
async def test_invalid_artists_fail_safely(client: LivePhishClient, response: object) -> None:
    """Schema changes produce a provider error instead of a raw parser failure."""
    with patch.object(client, "_request", new_callable=AsyncMock) as request:
        request.return_value = {"Response": response}
        with pytest.raises(LivePhishError, match="catalog:"):
            await client.artists()


async def test_unavailable_catalog_is_not_cached(client: LivePhishClient) -> None:
    """An unavailable catalog response must not poison the cache."""
    with patch.object(client, "_request", new_callable=AsyncMock) as request:
        request.return_value = {"responseAvailabilityCode": 1, "Response": {"artists": []}}
        with pytest.raises(LivePhishError, match="unavailable"):
            await client.artists()
        assert not client._catalog_cache


async def test_artist_releases_include_separate_album_catalog(client: LivePhishClient) -> None:
    """The albumTitle selector must be fetched even when the show catalog is empty."""
    with patch.object(client, "_request", new_callable=AsyncMock) as request:
        request.side_effect = [
            {"Response": {"containers": []}},
            {"Response": {"containers": [{"containerID": 2653, "containerType": 1}]}},
        ]
        assert await client.artist_releases("7") == [{"containerID": 2653, "containerType": 1}]
        assert request.await_args_list[0].kwargs["params"]["sortBy"] == "performanceDate"
        assert request.await_args_list[1].kwargs["params"]["sortBy"] == "albumTitle"


async def test_album_category_skips_show_requests(client: LivePhishClient) -> None:
    """Album-only enumeration must never fetch the concert catalog."""
    with patch.object(client, "_request", new_callable=AsyncMock) as request:
        request.return_value = {
            "Response": {"containers": [{"containerID": 2653, "containerType": 1}]}
        }
        assert await client.artist_releases("7", albums_only=True) == [
            {"containerID": 2653, "containerType": 1}
        ]
        request.assert_awaited_once()
        assert request.call_args.kwargs["params"]["sortBy"] == "albumTitle"


async def test_album_category_reads_every_page(client: LivePhishClient) -> None:
    """Album-only mode retains paging instead of truncating artists at 100 releases."""
    page = [{"containerID": number, "containerType": 1} for number in range(1, 101)]
    with patch.object(client, "releases", new_callable=AsyncMock) as releases:
        releases.side_effect = [page, [{"containerID": 101, "containerType": 1}]]
        assert len(await client.artist_releases("1", albums_only=True)) == 101
        assert [call.kwargs["offset"] for call in releases.call_args_list] == [0, 100]
        assert all(call.kwargs["albums_only"] for call in releases.call_args_list)


async def test_search_preserves_group_song_titles(client: LivePhishClient) -> None:
    """The service supplies song names on group headers, not individual performances."""
    first = {"containerID": 1747, "trackID": 29990}
    second = {"containerID": 1748, "trackID": 29991}
    with patch.object(client, "_request", new_callable=AsyncMock) as request:
        request.return_value = {
            "Response": {
                "catalogSearchTypeContainers": [
                    {
                        "matchType": 2,
                        "catalogSearchContainers": [
                            {"matchedStr": "Axilla", "catalogSearchResultItems": [first]},
                            {
                                "matchedStr": "Axilla (Part II)",
                                "catalogSearchResultItems": [second],
                            },
                        ],
                    }
                ]
            }
        }
        rows = await client.search("Axilla")
        assert [row["songTitle"] for row in rows] == ["Axilla", "Axilla (Part II)"]
        assert "songTitle" not in first
        assert "songTitle" not in second

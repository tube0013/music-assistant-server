"""Read-only saved playlist integration and navigation tests."""

from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from music_assistant_models.errors import MediaNotFoundError

from music_assistant.providers.livephish import LivePhishProvider
from music_assistant.providers.livephish.client import LivePhishClient, LivePhishError

PLAYLIST = {"id": "42", "name": "Saved mix", "trackCount": 2}
TRACK = {
    "id": "entry-1",
    "trackId": "29990",
    "releaseId": "1747",
    "name": "Axilla",
    "albumTitle": "Moon Palace",
    "durationSeconds": 455,
    "artist": {"id": 1, "name": "Phish"},
    "image": {"url": "https://images.example.test/cover.jpg"},
}


@pytest.fixture
def client() -> LivePhishClient:
    """Create a client without real network access or credentials."""
    client = LivePhishClient(MagicMock(), "", "")
    client._access_token = "synthetic-token"
    return client


async def test_playlists_auth_cache_and_pagination(client: LivePhishClient) -> None:
    """Use the LivePhish stash host, authenticate, and cache all returned pages."""
    with (
        patch.object(client, "authenticate", new_callable=AsyncMock) as auth,
        patch.object(client, "_request", new_callable=AsyncMock) as request,
    ):
        request.side_effect = [
            {"items": [PLAYLIST], "total": 2},
            {"items": [{**PLAYLIST, "id": "43"}], "total": 2},
        ]
        assert len(await client.playlists()) == 2
        assert len(await client.playlists()) == 2
        assert auth.await_count == 4
        assert request.await_count == 2
        assert request.await_args_list[0].args[:2] == (
            "GET",
            "https://stash.livephish.com/api/v1/me/playlists",
        )
        assert request.await_args_list[1].kwargs["params"]["offset"] == 1
        assert request.call_args.kwargs["headers"] == {"Authorization": "Bearer synthetic-token"}


async def test_playlist_failure_is_not_cached(client: LivePhishClient) -> None:
    """Account API failures propagate safely instead of looking like an empty library."""
    with (
        patch.object(client, "authenticate", new_callable=AsyncMock),
        patch.object(client, "_request", new_callable=AsyncMock) as request,
    ):
        request.side_effect = LivePhishError("playlists: HTTP 401")
        with pytest.raises(LivePhishError, match="401"):
            await client.playlists()
        assert not client._stash_cache


async def test_playlist_repeated_page_is_rejected(client: LivePhishClient) -> None:
    """Stop if the account API ignores an offset instead of looping forever."""
    with patch.object(client, "_stash", new_callable=AsyncMock) as request:
        request.return_value = {"items": [PLAYLIST], "total": 2}
        with pytest.raises(LivePhishError, match="pagination did not advance"):
            await client.playlists()


@pytest.mark.parametrize("playlist_id", ["", "../42", "42?query", "42/tracks"])
async def test_playlist_id_validation(client: LivePhishClient, playlist_id: str) -> None:
    """Reject malformed IDs before constructing authenticated account requests."""
    with pytest.raises(LivePhishError, match="invalid playlist ID"):
        await client.playlist(playlist_id)


async def test_playlist_details_are_direct_objects(client: LivePhishClient) -> None:
    """LivePhish returns the playlist directly rather than under an items key."""
    with patch.object(client, "_stash", new_callable=AsyncMock, return_value=PLAYLIST):
        assert await client.playlist("42") == PLAYLIST


async def test_saved_playlist_browse_and_library(provider: LivePhishProvider) -> None:
    """Saved playlists are discoverable through both browsing and library sync."""
    client = cast("MagicMock", provider._client)
    client.playlists.return_value = [PLAYLIST]
    client.playlist.return_value = PLAYLIST
    browse = await provider.browse("livephish--test://playlists")
    library = [item async for item in provider.get_library_playlists()]
    assert [item.item_id for item in browse] == ["42"]
    assert library[0].is_editable is False
    assert (await provider.get_playlist("42")).name == "Saved mix"


async def test_playlist_order_duplicates_and_playback_ids(provider: LivePhishProvider) -> None:
    """Repeated performances retain their positions and composite playback IDs."""
    client = cast("MagicMock", provider._client)
    client.playlist_tracks.return_value = [TRACK, {**TRACK, "id": "entry-2"}]
    tracks = await provider.get_playlist_tracks("42")
    assert [item.position for item in tracks] == [1, 2]
    assert [item.item_id for item in tracks] == ["1747:29990", "1747:29990"]
    assert tracks[0].duration == 455
    assert tracks[0].album
    assert tracks[0].album.item_id == "1747"
    assert await provider.get_playlist_tracks("42", page=1) == []
    client.playlist_tracks.assert_awaited_once_with("42")


async def test_playlist_missing_show_id_fails_safely(provider: LivePhishProvider) -> None:
    """Never invent a show ID when a playlist row cannot be played."""
    client = cast("MagicMock", provider._client)
    client.playlist_tracks.return_value = [{**TRACK, "releaseId": None}]
    with pytest.raises(MediaNotFoundError, match="missing catalog metadata"):
        await provider.get_playlist_tracks("42")


@pytest.mark.parametrize(
    ("method", "endpoint"),
    [
        ("favorite_albums", "releases/favorite"),
    ],
)
async def test_favorites_use_stash_pagination(
    client: LivePhishClient, method: str, endpoint: str
) -> None:
    """Both saved-library resources share authenticated, cached stash paging."""
    with patch.object(client, "_stash", new_callable=AsyncMock) as request:
        request.side_effect = [
            {"items": [{"id": 1}], "total": 2},
            {"items": [{"id": 2}], "total": 2},
        ]
        assert len(await getattr(client, method)()) == 2
        assert request.call_args_list[0].args == (endpoint,)
        assert request.call_args_list[1].kwargs["offset"] == 1


async def test_playlist_page_refreshes_expired_token(client: LivePhishClient) -> None:
    """The next uncached page must use the token refreshed during authentication."""

    async def authenticate() -> None:
        client._access_token = "refreshed-token"

    with (
        patch.object(client, "authenticate", new_callable=AsyncMock, side_effect=authenticate),
        patch.object(client, "_request", new_callable=AsyncMock) as request,
    ):
        request.return_value = {"items": [], "total": 0}
        await client._stash("playlists", limit=100, offset=100)
        assert request.call_args.kwargs["headers"] == {"Authorization": "Bearer refreshed-token"}

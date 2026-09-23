"""LivePhish catalog library enumeration tests."""

from typing import cast
from unittest.mock import MagicMock

import pytest
from music_assistant_models.enums import AlbumType, MediaType, ProviderFeature
from music_assistant_models.errors import ProviderUnavailableError

from music_assistant.models.music_provider import SYNC_RUN_STATE, SyncRunState
from music_assistant.providers.livephish import LivePhishProvider
from music_assistant.providers.livephish.client import LivePhishError
from tests.providers.livephish.fixtures import RELEASES


def test_no_saved_artist_support(provider: LivePhishProvider) -> None:
    """LivePhish has no saved-artist library; catalog artists remain browsable."""
    assert ProviderFeature.LIBRARY_ARTISTS not in provider.supported_features
    assert ProviderFeature.LIBRARY_ALBUMS in provider.supported_features
    assert ProviderFeature.LIBRARY_PLAYLISTS in provider.supported_features


async def test_saved_library_albums_without_catalog_requests(
    provider: LivePhishProvider,
) -> None:
    """Saved releases alone populate the library without fetching the full catalog."""
    client = cast("MagicMock", provider._client)
    client.favorite_albums.return_value = [
        {"id": 42, "title": "Saved Album", "artist": {"id": 1, "name": "Phish"}}
    ]
    albums = [item async for item in provider.get_library_albums()]
    assert [album.item_id for album in albums] == ["42"]
    assert albums[0].name == "Saved Album"
    client.favorite_albums.assert_awaited_once()
    assert ProviderFeature.LIBRARY_ALBUMS in provider.supported_features
    client.artists.assert_not_awaited()
    client.artist_releases.assert_not_awaited()
    client.album.assert_not_awaited()


def test_playlist_images_are_normalized_before_serialization(provider: LivePhishProvider) -> None:
    """Even direct consumers receive the corrected image URL before provider startup."""
    image = "https://www.livephish.com/assets/phish/images/pix/shows/ph260906_01.jpg"
    playlist = provider._parse_playlist({"id": "1", "name": "Mix", "imageUrl": image})
    assert playlist.metadata.images
    assert playlist.metadata.images[0].path.startswith("https://s3.amazonaws.com/static.nugs.net/")
    track = provider._parse_playlist_track(
        {
            "trackId": "41128",
            "releaseId": "2770",
            "name": "Light",
            "artist": {"id": 1, "name": "Phish"},
            "image": {"url": image},
        },
        1,
    )
    assert track.metadata.images
    assert track.metadata.images[0].path == playlist.metadata.images[0].path


def test_browsed_show_keeps_live_type(provider: LivePhishProvider) -> None:
    """Show metadata remains available as a live album outside automatic sync."""
    assert provider._parse_album(RELEASES[0]).album_type == AlbumType.LIVE


async def test_saved_album_failure_is_not_empty_library(provider: LivePhishProvider) -> None:
    """An upstream favorites outage cannot masquerade as deletions."""
    cast("MagicMock", provider._client).favorite_albums.side_effect = LivePhishError(
        "stash: HTTP 500"
    )
    with pytest.raises(ProviderUnavailableError, match="500"):
        _ = [item async for item in provider.get_library_albums()]


def test_catalog_media_types_are_available_for_matching(provider: LivePhishProvider) -> None:
    """Search-based matching includes tracks and artists without library enumeration."""
    assert provider.supported_media_types == {
        MediaType.ARTIST,
        MediaType.ALBUM,
        MediaType.TRACK,
        MediaType.PLAYLIST,
    }
    assert ProviderFeature.LIBRARY_TRACKS not in provider.supported_features
    assert ProviderFeature.LIBRARY_ARTISTS not in provider.supported_features


@pytest.mark.parametrize("media_type", [MediaType.ALBUM, MediaType.PLAYLIST])
@pytest.mark.parametrize("identified", [True, False])
async def test_bad_library_item_preserves_sync_state_and_continues(
    provider: LivePhishProvider, media_type: MediaType, identified: bool
) -> None:
    """A bad entry protects its old mapping (or all deletions) without hiding later items."""
    client = cast("MagicMock", provider._client)
    bad = {"id": "42", "artist": {"name": "Phish"}} if identified else {}
    if media_type == MediaType.ALBUM:
        client.favorite_albums.return_value = [
            bad,
            {"id": "43", "title": "Good", "artist": {"id": 1, "name": "Phish"}},
        ]
    else:
        client.playlists.return_value = [bad, {"id": "43", "name": "Good"}]
    listing = (
        provider.get_library_albums()
        if media_type == MediaType.ALBUM
        else provider.get_library_playlists()
    )
    state = SyncRunState()
    token = SYNC_RUN_STATE.set(state)
    try:
        assert [item.item_id async for item in listing] == ["43"]
        assert state.failures == 1
        if identified:
            assert state.skipped_item_ids[media_type] == {"42"}
            assert not state.incomplete_media_types
        else:
            assert state.incomplete_media_types == {media_type}
    finally:
        SYNC_RUN_STATE.reset(token)


async def test_saved_playlist_outage_aborts_listing(provider: LivePhishProvider) -> None:
    """Transport errors abort sync instead of being treated as missing playlists."""
    cast("MagicMock", provider._client).playlists.side_effect = LivePhishError("stash: HTTP 500")
    with pytest.raises(ProviderUnavailableError, match="500"):
        _ = [item async for item in provider.get_library_playlists()]

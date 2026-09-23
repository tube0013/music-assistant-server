"""Exercise multi-artist browsing and search through the real provider."""

from typing import cast
from unittest.mock import MagicMock

import pytest
from music_assistant_models.enums import MediaType
from music_assistant_models.errors import MediaNotFoundError
from music_assistant_models.media_items import Album, BrowseFolder, Track

from music_assistant.providers.livephish import LivePhishProvider
from tests.providers.livephish.fixtures import RELEASES


async def test_browse_artist_year_show(provider: LivePhishProvider) -> None:
    """Side projects expose only their actual release years and shows."""
    root = await provider.browse("livephish--test://")
    assert [item.item_id for item in root] == ["artists", "releases", "albums", "playlists"]
    assert isinstance(root[0], BrowseFolder)
    artists = await provider.browse(root[0].path)
    assert [item.name for item in artists] == ["Phish", "Oysterhead"]
    assert isinstance(artists[1], BrowseFolder)
    years = await provider.browse(artists[1].path)
    assert [item.item_id for item in years] == [
        "artists/19/all",
        "artists/19/albums",
        "artists/19/2022",
        "artists/19/2020",
    ]
    assert isinstance(years[2], BrowseFolder)
    shows = await provider.browse(years[2].path)
    assert len(shows) == 1
    assert isinstance(shows[0], Album)
    assert shows[0].item_id == "1773"
    assert shows[0].artists[0].name == "Oysterhead"
    assert shows[0].artists[0].item_id == "19"


async def test_artist_albums_keep_one_artist_identity(provider: LivePhishProvider) -> None:
    """All releases from one artist share a stable service artist ID."""
    albums = await provider.get_artist_albums("19")
    assert len(albums) == 2
    assert {item.artists[0].item_id for item in albums} == {"19"}
    assert albums[0].name == "05/01/22 Sweetwater 420 Fest"
    assert albums[0].year == 2022
    assert albums[0].metadata.images
    assert (
        albums[0].metadata.images[0].path
        == "https://s3.amazonaws.com/static.nugs.net/assets/test.jpg"
    )


async def test_test_configuration_is_not_exposed(provider: LivePhishProvider) -> None:
    """Remove the temporary configured-show entry from browsing and settings."""
    assert all(item.item_id != "configured" for item in await provider.browse("livephish--test://"))
    assert all(entry.key != "show_ids" for entry in await provider.get_config_entries())


@pytest.mark.parametrize("path", ["releases", "releases/0", "releases/"])
async def test_latest_shows_parent_path(provider: LivePhishProvider, path: str) -> None:
    """The parent route from the recording resolves without a retry loop."""
    cast("MagicMock", provider._client).releases.return_value = [
        {**RELEASES[0], "containerID": number} for number in range(1, 101)
    ]
    items = await provider.browse(f"livephish--test://{path}")
    assert len(items) == 101
    assert all(isinstance(item, Album) for item in items[:-1])
    assert isinstance(items[-1], BrowseFolder)
    assert items[-1].path.endswith("releases/100")


async def test_albums_grouped_by_artist(provider: LivePhishProvider) -> None:
    """Albums open artist folders, then the artist's complete album listing."""
    folders = await provider.browse("livephish--test://albums")
    assert [folder.item_id for folder in folders] == ["albums/1"]
    catalog = cast("MagicMock", provider._client)
    catalog.artist_releases.return_value = [
        {**RELEASES[0], "containerID": i, "containerType": 1, "artistID": 1} for i in range(1, 105)
    ]
    albums = await provider.browse("livephish--test://albums/1")
    assert len(albums) == 104
    assert all(isinstance(item, Album) for item in albums)


@pytest.mark.parametrize(
    "path", ["wrong://artists", "livephish--test://artists/x", "livephish--test://releases/-1"]
)
async def test_invalid_browse_paths(provider: LivePhishProvider, path: str) -> None:
    """Malformed paths fail before catalog access."""
    with pytest.raises(MediaNotFoundError):
        await provider.browse(path)


async def test_artist_search_does_not_fetch_shows(provider: LivePhishProvider) -> None:
    """Honor requested media types and case-insensitive artist search."""
    result = await provider.search("OYSTER", [MediaType.ARTIST])
    assert [item.item_id for item in result.artists] == ["19"]
    cast("MagicMock", provider._client).search.assert_not_awaited()


async def test_search_deduplicates_and_resolves_tracks(provider: LivePhishProvider) -> None:
    """Build search matches without per-show lookups, keeping restart-safe IDs."""
    item = {**RELEASES[0], "trackID": 29990, "songTitle": "Test song", "totalRunningTime": "454.5"}
    cast("MagicMock", provider._client).search.return_value = [item, item]
    cast("MagicMock", provider._client).album.return_value = {
        **RELEASES[0],
        "tracks": [{"trackID": 29990, "songTitle": "Test song", "totalRunningTime": 454}],
    }
    result = await provider.search("Test song", [MediaType.ALBUM, MediaType.TRACK], limit=1)
    assert len(result.albums) == len(result.tracks) == 1
    assert result.tracks[0].item_id == "1773:29990"
    assert isinstance(result.tracks[0], Track)
    assert result.tracks[0].duration == 454
    assert result.tracks[0].artists[0].name == "Oysterhead"
    cast("MagicMock", provider._client).album.assert_not_awaited()


async def test_track_exclusions_and_disc_order(provider: LivePhishProvider) -> None:
    """Catalog expansion preserves exclusion rules and multi-disc running order."""
    cast("MagicMock", provider._client).album.return_value = {
        **RELEASES[0],
        "tracks": [
            {"trackID": 2, "discNum": 2, "trackNum": 1},
            {"trackID": 3, "trackExclude": 1},
            {"trackID": 1, "discNum": 1, "trackNum": 2},
        ],
    }
    assert [item.item_id for item in await provider.get_album_tracks("1773")] == [
        "1773:1",
        "1773:2",
    ]


async def test_album_only_artist_has_browseable_releases(provider: LivePhishProvider) -> None:
    """Studio albums remain discoverable for artists with no live shows."""
    catalog = cast("MagicMock", provider._client)
    catalog.artists.return_value = [{"artistID": 7, "artistName": "Page McConnell"}]
    catalog.artist_releases.return_value = [
        {
            "containerID": 2653,
            "containerType": 1,
            "containerInfo": "Something Will Land",
            "artistID": 7,
            "artistName": "Page McConnell",
        }
    ]
    folders = await provider.browse("livephish--test://artists/7")
    assert [item.item_id for item in folders] == ["artists/7/all", "artists/7/albums"]
    items = await provider.browse("livephish--test://artists/7/albums")
    assert len(items) == 1
    assert items[0].name == "Something Will Land"


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        (
            "/assets/phish/images/pix/shows/ph260906_01.jpg",
            "https://s3.amazonaws.com/static.nugs.net/assets/phish/images/pix/shows/ph260906_01.jpg",
        ),
        (
            "assets/phish/images/pix/shows/ph260906_01.jpg",
            "https://s3.amazonaws.com/static.nugs.net/assets/phish/images/pix/shows/ph260906_01.jpg",
        ),
        (
            "https://images.example.test/cover.jpg?size=large",
            "https://images.example.test/cover.jpg?size=large",
        ),
        ("http://images.example.test/cover.jpg", "http://images.example.test/cover.jpg"),
        ("//images.example.test/cover.jpg", "https://images.example.test/cover.jpg"),
    ],
)
def test_artwork_uses_legacy_cdn(provider: LivePhishProvider, path: str, expected: str) -> None:
    """Resolve legacy relative images without rewriting absolute image URLs."""
    album = provider._parse_album({**RELEASES[0], "img": {"url": path}})
    assert album.metadata.images
    assert album.metadata.images[0].path == expected


@pytest.mark.parametrize("image", [None, {}, {"url": ""}, {"url": "file:///cover.jpg"}])
def test_unusable_artwork_is_omitted(provider: LivePhishProvider, image: object) -> None:
    """Missing or unsupported artwork must not prevent catalog parsing."""
    album = provider._parse_album({**RELEASES[0], "img": image})
    assert not album.metadata.images


async def test_track_search_25_shows_requires_no_album_requests(
    provider: LivePhishProvider,
) -> None:
    """The default MA search limit cannot trigger 25 sequential detail fetches."""
    client = cast("MagicMock", provider._client)
    client.search.return_value = [
        {
            **RELEASES[0],
            "containerID": number,
            "trackID": number + 100,
            "songTitle": "Axilla",
            "availability": 1,
        }
        for number in range(1, 31)
    ]
    result = await provider.search("Axilla", [MediaType.TRACK], limit=25)
    assert len(result.tracks) == 25
    assert all(item.name == "Axilla" for item in result.tracks)
    assert all(isinstance(item, Track) and item.duration == 0 for item in result.tracks)
    assert (
        len(
            {item.album.item_id for item in result.tracks if isinstance(item, Track) and item.album}
        )
        == 25
    )
    client.search.assert_awaited_once_with("Axilla")
    client.album.assert_not_awaited()
    client.artists.assert_not_awaited()


async def test_search_skips_incomplete_track_matches(provider: LivePhishProvider) -> None:
    """Missing song titles or artist IDs must not produce mislabeled results or lookups."""
    client = cast("MagicMock", provider._client)
    client.search.return_value = [
        {**RELEASES[0], "trackID": 1},
        {**RELEASES[0], "artistID": None, "trackID": 2, "songTitle": "Invalid"},
        {**RELEASES[0], "trackID": 3, "songTitle": "Valid", "totalRunningTime": "bad"},
    ]
    result = await provider.search("song", [MediaType.TRACK])
    assert [item.name for item in result.tracks] == ["Valid"]
    assert isinstance(result.tracks[0], Track)
    assert result.tracks[0].duration == 0
    client.album.assert_not_awaited()


@pytest.mark.parametrize(
    ("duration", "expected"), [("123.4", 123), (None, 0), ("bad", 0), ("inf", 0), (-1, 0)]
)
async def test_album_track_durations(
    provider: LivePhishProvider, duration: object, expected: int
) -> None:
    """Malformed and fractional service durations cannot abort an album."""
    cast("MagicMock", provider._client).album.return_value = {
        **RELEASES[0],
        "tracks": [{"trackID": 1, "totalRunningTime": duration}],
    }
    tracks = await provider.get_album_tracks("1773")
    assert tracks[0].duration == expected


async def test_latest_next_page(provider: LivePhishProvider) -> None:
    """A final short page does not offer a dead next-page link."""
    client = cast("MagicMock", provider._client)
    client.releases.return_value = [RELEASES[0]]
    items = await provider.browse("livephish--test://releases/100")
    assert len(items) == 1
    client.releases.assert_awaited_once_with(offset=100)

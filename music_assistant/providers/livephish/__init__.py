"""LivePhish music source for Music Assistant."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from urllib.parse import urljoin, urlsplit

from music_assistant_models.config_entries import ConfigEntry, ConfigValueOption
from music_assistant_models.enums import (
    AlbumType,
    ConfigEntryType,
    ContentType,
    ImageType,
    MediaType,
    ProviderFeature,
    StreamType,
)
from music_assistant_models.errors import (
    AudioError,
    LoginFailed,
    MediaNotFoundError,
    ProviderUnavailableError,
)
from music_assistant_models.media_items import (
    Album,
    Artist,
    AudioFormat,
    BrowseFolder,
    ItemMapping,
    MediaItemImage,
    Playlist,
    ProviderMapping,
    SearchResults,
    Track,
    UniqueList,
)
from music_assistant_models.streamdetails import StreamDetails

from music_assistant.constants import CONF_ENTRY_UNOFFICIAL_PROVIDER, CONF_PASSWORD, CONF_USERNAME
from music_assistant.models.music_provider import MusicProvider

from .client import USER_AGENT, LivePhishAuthError, LivePhishClient, LivePhishError

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Sequence

    from music_assistant_models.config_entries import ProviderConfig
    from music_assistant_models.provider import ProviderManifest

    from music_assistant.mass import MusicAssistant


IMAGE_BASE = "https://s3.amazonaws.com/static.nugs.net/"

SUPPORTED_FEATURES = {
    ProviderFeature.BROWSE,
    ProviderFeature.SEARCH,
    ProviderFeature.ARTIST_ALBUMS,
    ProviderFeature.LIBRARY_PLAYLISTS,
    ProviderFeature.LIBRARY_ALBUMS,
}


async def setup(
    mass: MusicAssistant, manifest: ProviderManifest, config: ProviderConfig
) -> LivePhishProvider:
    """Create the provider."""
    return LivePhishProvider(mass, manifest, config, SUPPORTED_FEATURES)


class LivePhishProvider(MusicProvider):
    """Browse the multi-artist LivePhish catalog and play entitled audio through MA."""

    @property
    def supported_media_types(self) -> set[MediaType]:
        """Return the media types available for browsing, search, and playback."""
        return {MediaType.ARTIST, MediaType.ALBUM, MediaType.TRACK, MediaType.PLAYLIST}

    @property
    def max_concurrent_streams(self) -> int:
        """Apply LivePhish's single-stream account limit."""
        return 1

    async def get_config_entries(self) -> tuple[ConfigEntry, ...]:
        """Return runtime options."""
        return (
            CONF_ENTRY_UNOFFICIAL_PROVIDER,
            ConfigEntry(
                key="quality",
                type=ConfigEntryType.STRING,
                default_value="auto",
                options=[ConfigValueOption(value) for value in ("auto", "flac", "alac", "aac")],
            ),
        )

    async def handle_async_init(self) -> None:
        """Validate credentials and prepare a paced API client."""
        self._client = LivePhishClient(
            self.mass.http_session,
            str(self.get_setup_value(CONF_USERNAME) or ""),
            str(self.get_setup_value(CONF_PASSWORD) or ""),
        )
        try:
            await self._client.authenticate()
        except LivePhishAuthError as err:
            raise LoginFailed(str(err)) from None
        except LivePhishError as err:
            raise ProviderUnavailableError(str(err)) from None

    async def browse(self, path: str) -> Sequence[Album | Playlist | BrowseFolder]:
        """Browse artists, release years, albums by artist, and saved playlists."""
        prefix, separator, subpath = path.partition("://")
        if not separator or prefix not in (self.instance_id, self.domain):
            raise MediaNotFoundError("Invalid LivePhish browse path")
        subpath = subpath.strip("/")
        parts = subpath.split("/") if subpath else []
        try:
            if not parts:
                return [
                    self._folder("artists", "artists"),
                    self._folder("releases", "recent_releases"),
                    self._folder("albums", "albums"),
                    self._folder("playlists", "playlists"),
                ]
            if parts == ["playlists"]:
                return [self._parse_playlist(item) for item in await self._client.playlists()]
            if parts in (["artists"], ["albums"]):
                albums_only = parts[0] == "albums"
                return [
                    self._folder(
                        f"albums/{item['artistID']}"
                        if albums_only
                        else f"artists/{item['artistID']}",
                        name=item["artistName"],
                    )
                    for item in await self._client.artists()
                    if not albums_only or item.get("numAlbums", 0)
                ]
            if parts == ["releases"] or (
                len(parts) == 2 and parts[0] == "releases" and parts[1].isdecimal()
            ):
                offset = int(parts[1]) if len(parts) == 2 else 0
                items = await self._client.releases(offset=offset)
                results: list[Album | BrowseFolder] = [self._parse_album(item) for item in items]
                if len(items) == 100:
                    results.append(self._folder(f"releases/{offset + len(items)}", "next_page"))
                return results
            if len(parts) == 2 and parts[0] == "albums" and parts[1].isdecimal():
                await self.get_artist(parts[1])
                items = await self._client.artist_releases(parts[1])
                return [self._parse_album(item) for item in items if item.get("containerType") == 1]
            if len(parts) in (2, 3) and parts[0] == "artists" and parts[1].isdecimal():
                await self.get_artist(parts[1])
                items = await self._client.artist_releases(parts[1])
                if len(parts) == 2:
                    years = sorted(
                        {self._year(item) for item in items if item.get("containerType") != 1}
                        - {0},
                        reverse=True,
                    )
                    return [
                        self._folder(f"{subpath}/all", "all_releases"),
                        self._folder(f"{subpath}/albums", "albums"),
                    ] + [self._folder(f"{subpath}/{year}", name=str(year)) for year in years]
                if parts[2] == "all":
                    return [self._parse_album(item) for item in items]
                if parts[2] == "albums":
                    return [
                        self._parse_album(item) for item in items if item.get("containerType") == 1
                    ]
                if parts[2].isdecimal():
                    return [
                        self._parse_album(item)
                        for item in items
                        if self._year(item) == int(parts[2]) and item.get("containerType") != 1
                    ]
        except LivePhishError as err:
            raise MediaNotFoundError(str(err)) from None
        raise MediaNotFoundError("Invalid LivePhish browse path")

    async def search(
        self, search_query: str, media_types: list[MediaType], limit: int = 5
    ) -> SearchResults:
        """Search artists, releases, and individual performances of songs."""
        result = SearchResults()
        if not search_query.strip() or limit <= 0:
            return result
        try:
            if MediaType.ARTIST in media_types:
                result.artists = [
                    self._parse_artist(item)
                    for item in await self._client.artists()
                    if search_query.casefold().strip() in item["artistName"].casefold()
                ][:limit]
            if not {MediaType.ALBUM, MediaType.TRACK}.intersection(media_types):
                return result
            items = await self._client.search(search_query)
        except LivePhishError as err:
            raise MediaNotFoundError(str(err)) from None
        albums: dict[str, Album] = {}
        tracks: dict[str, Track] = {}
        for item in items:
            if not item.get("artistID") or not item.get("artistName"):
                continue
            album_id = str(item["containerID"])
            album = self._parse_album(item)
            if MediaType.ALBUM in media_types and len(albums) < limit:
                albums.setdefault(album_id, album)
            if MediaType.TRACK in media_types and item.get("trackID") not in (None, 0, "0"):
                track_id = f"{album_id}:{item['trackID']}"
                if track_id not in tracks and len(tracks) < limit and item.get("songTitle"):
                    tracks[track_id] = self._parse_search_track(item, album)
            if (MediaType.ALBUM not in media_types or len(albums) >= limit) and (
                MediaType.TRACK not in media_types or len(tracks) >= limit
            ):
                break
        result.albums = list(albums.values())
        result.tracks = list(tracks.values())
        return result

    async def get_artist(self, prov_artist_id: str) -> Artist:
        """Resolve an artist by its stable catalog ID."""
        try:
            for item in await self._client.artists():
                if str(item["artistID"]) == prov_artist_id:
                    return self._parse_artist(item)
        except LivePhishError as err:
            raise MediaNotFoundError(str(err)) from None
        raise MediaNotFoundError("LivePhish artist not found")

    async def get_artist_albums(self, prov_artist_id: str) -> list[Album]:
        """Get every show and studio release for an artist."""
        await self.get_artist(prov_artist_id)
        try:
            return [
                self._parse_album(item)
                for item in await self._client.artist_releases(prov_artist_id)
            ]
        except LivePhishError as err:
            raise MediaNotFoundError(str(err)) from None

    async def get_album(self, prov_album_id: str) -> Album:
        """Get an album by LivePhish container ID."""
        data = await self._album_data(prov_album_id)
        return self._parse_album(data)

    async def get_album_tracks(self, prov_album_id: str) -> list[Track]:
        """Get tracks in disc and track order."""
        data = await self._album_data(prov_album_id)
        album = self._parse_album(data)
        tracks = []
        for raw in data["tracks"]:
            if not raw.get("trackID") or raw.get("trackExclude") in (1, "1", True):
                continue
            item_id = f"{prov_album_id}:{raw['trackID']}"
            tracks.append(
                Track(
                    item_id=item_id,
                    provider=self.instance_id,
                    name=raw.get("songTitle") or "Untitled",
                    provider_mappings=self._mappings(item_id),
                    artists=album.artists,
                    album=ItemMapping.from_item(album),
                    duration=self._duration(raw.get("totalRunningTime")),
                    disc_number=int(raw.get("discNum") or 1),
                    track_number=int(raw.get("trackNum") or 0),
                )
            )
        return sorted(tracks, key=lambda track: (track.disc_number, track.track_number))

    async def get_track(self, prov_track_id: str) -> Track:
        """Resolve tracks after restarts using a composite show:track ID."""
        album_id, separator, _ = prov_track_id.partition(":")
        if separator:
            for track in await self.get_album_tracks(album_id):
                if track.item_id == prov_track_id:
                    return track
        raise MediaNotFoundError("LivePhish track not found")

    async def get_library_albums(self) -> AsyncGenerator[Album]:
        """Import only the account's saved releases, matching Nugs library behavior."""
        try:
            for item in await self._client.favorite_albums():
                try:
                    album = self._parse_album(item)
                except LivePhishError as err:
                    raw_id = item.get("releaseId") or item.get("id") or item.get("containerID")
                    self.report_skipped_sync_item(
                        MediaType.ALBUM, str(raw_id) if raw_id else None, err
                    )
                    continue
                yield album
        except LivePhishError as err:
            raise ProviderUnavailableError(str(err)) from None

    async def get_library_playlists(self) -> AsyncGenerator[Playlist]:
        """Import the account's saved playlists without modifying LivePhish."""
        try:
            for item in await self._client.playlists():
                try:
                    playlist = self._parse_playlist(item)
                except LivePhishError as err:
                    raw_id = item.get("id")
                    self.report_skipped_sync_item(
                        MediaType.PLAYLIST, str(raw_id) if raw_id else None, err
                    )
                    continue
                yield playlist
        except LivePhishError as err:
            raise ProviderUnavailableError(str(err)) from None

    async def get_playlist(self, prov_playlist_id: str) -> Playlist:
        """Resolve a saved playlist by its LivePhish ID."""
        try:
            return self._parse_playlist(await self._client.playlist(prov_playlist_id))
        except LivePhishError as err:
            raise MediaNotFoundError(str(err)) from None

    async def get_playlist_tracks(self, prov_playlist_id: str, page: int = 0) -> list[Track]:
        """Return saved performances in playlist order with restart-safe track IDs."""
        if page > 0:
            return []
        try:
            items = await self._client.playlist_tracks(prov_playlist_id)
            return [
                self._parse_playlist_track(item, position) for position, item in enumerate(items, 1)
            ]
        except LivePhishError as err:
            raise MediaNotFoundError(str(err)) from None

    async def get_stream_details(self, item_id: str, media_type: MediaType) -> StreamDetails:
        """Resolve audio just before playback; keep its URL inside the server."""
        if media_type != MediaType.TRACK:
            raise MediaNotFoundError("Only tracks are supported")
        track = await self.get_track(item_id)
        try:
            url = await self._client.stream(
                item_id.split(":", 1)[1],
                quality=str(self.config.get_value("quality") or "auto"),
            )
        except LivePhishError as err:
            raise AudioError(str(err)) from None
        return StreamDetails(
            item_id=item_id,
            provider=self.instance_id,
            duration=track.duration,
            audio_format=AudioFormat(content_type=ContentType.UNKNOWN),
            stream_type=StreamType.HTTP,
            can_seek=True,
            allow_seek=True,
            path=url,
            extra_input_args=["-user_agent", USER_AGENT, "-referer", "https://plus.livephish.com/"],
        )

    async def _album_data(self, album_id: str) -> dict[str, Any]:
        try:
            return await self._client.album(album_id)
        except LivePhishError as err:
            raise MediaNotFoundError(str(err)) from None

    def _duration(self, value: object) -> int:
        try:
            return max(0, int(float(str(value or 0))))
        except TypeError, ValueError, OverflowError:
            return 0

    def _parse_search_track(self, item: dict[str, Any], album: Album) -> Track:
        item_id = f"{album.item_id}:{item['trackID']}"
        duration = self._duration(item.get("totalRunningTime"))
        track = Track(
            item_id=item_id,
            provider=self.instance_id,
            name=item["songTitle"],
            provider_mappings=self._mappings(item_id),
            artists=album.artists,
            album=ItemMapping.from_item(album),
            duration=duration,
        )
        if album.metadata.images:
            track.metadata.images = album.metadata.images
        return track

    def _mappings(self, item_id: str) -> set[ProviderMapping]:
        return {
            ProviderMapping(
                item_id=item_id, provider_domain=self.domain, provider_instance=self.instance_id
            )
        }

    def _parse_playlist(self, item: dict[str, Any]) -> Playlist:
        if not item.get("id") or not item.get("name"):
            raise LivePhishError("playlists: invalid playlist metadata")
        item_id = str(item["id"])
        playlist = Playlist(
            item_id=item_id,
            provider=self.instance_id,
            name=item["name"],
            provider_mappings=self._mappings(item_id),
            is_editable=False,
        )
        if image := self._image_url(item.get("imageUrl")):
            playlist.metadata.add_image(
                MediaItemImage(
                    type=ImageType.THUMB,
                    path=image,
                    provider=self.instance_id,
                    remotely_accessible=True,
                )
            )
        return playlist

    def _parse_playlist_track(self, item: dict[str, Any], position: int) -> Track:
        artist = item.get("artist")
        if (
            not item.get("trackId")
            or not item.get("releaseId")
            or not item.get("name")
            or not isinstance(artist, dict)
            or not artist.get("id")
            or not artist.get("name")
        ):
            raise LivePhishError("playlists: track is missing catalog metadata")
        album_id = str(item["releaseId"])
        item_id = f"{album_id}:{item['trackId']}"
        try:
            duration = int(float(item.get("durationSeconds") or 0))
        except ValueError, TypeError, OverflowError:
            raise LivePhishError("playlists: invalid track duration") from None
        track = Track(
            item_id=item_id,
            provider=self.instance_id,
            name=item["name"],
            provider_mappings=self._mappings(item_id),
            artists=UniqueList(
                [self._parse_artist({"artistID": artist["id"], "artistName": artist["name"]})]
            ),
            album=ItemMapping(
                media_type=MediaType.ALBUM,
                item_id=album_id,
                provider=self.instance_id,
                name=item.get("albumTitle") or album_id,
            ),
            duration=duration,
            position=position,
        )
        image = item.get("image")
        if isinstance(image, dict) and (image_url := self._image_url(image.get("url"))):
            track.metadata.add_image(
                MediaItemImage(
                    type=ImageType.THUMB,
                    path=image_url,
                    provider=self.instance_id,
                    remotely_accessible=True,
                )
            )
        return track

    def _folder(self, path: str, key: str = "", name: str = "") -> BrowseFolder:
        return BrowseFolder(
            item_id=path,
            provider=self.instance_id,
            path=f"{self.instance_id}://{path}",
            name=name,
            translation_key=key or None,
        )

    def _parse_artist(self, data: dict[str, Any]) -> Artist:
        raw_id = data.get("id") or data.get("artistID")
        if not raw_id:
            raise LivePhishError("catalog: artist ID missing")
        artist_id = str(raw_id)
        return Artist(
            item_id=artist_id,
            provider=self.instance_id,
            name=data.get("name") or data.get("artistName") or "Unknown artist",
            provider_mappings=self._mappings(artist_id),
        )

    def _year(self, data: dict[str, Any]) -> int:
        year = str(data.get("performanceDateYear") or "")
        if year.isdecimal():
            return int(year)
        date = str(data.get("performanceDate") or "")
        if "/" in date and date.rsplit("/", maxsplit=1)[-1].isdecimal():
            return int(date.rsplit("/", maxsplit=1)[-1])
        return 0

    def _parse_album(self, data: dict[str, Any]) -> Album:
        raw_id = data.get("releaseId") or data.get("id") or data.get("containerID")
        if not raw_id:
            raise LivePhishError("catalog: album ID missing")
        item_id = str(raw_id)
        title = data.get("title") or data.get("containerInfo") or data.get("containerName")
        if not title:
            date = data.get("performanceDateFormatted") or data.get("performanceDate") or ""
            title = f"{date} {data.get('venueName') or item_id}"
        album = Album(
            item_id=item_id,
            provider=self.instance_id,
            name=title.strip(),
            year=self._year(data) or None,
            album_type=AlbumType.ALBUM if str(data.get("containerType")) == "1" else AlbumType.LIVE,
            provider_mappings=self._mappings(item_id),
            artists=UniqueList([self._parse_artist(data.get("artist") or data)]),
        )
        image = data.get("image") or data.get("img")
        if isinstance(image, dict) and (image_url := self._image_url(image.get("url"))):
            album.metadata.add_image(
                MediaItemImage(
                    type=ImageType.THUMB,
                    path=image_url,
                    provider=self.instance_id,
                    remotely_accessible=True,
                )
            )
        return album

    def _image_url(self, path: object) -> str | None:
        if not isinstance(path, str) or not path:
            return None
        try:
            parsed = urlsplit(path)
            if (
                parsed.scheme in ("http", "https")
                and parsed.hostname in ("www.livephish.com", "livephish.com")
                and parsed.path.startswith("/assets/phish/")
            ):
                return parsed._replace(
                    scheme="https",
                    netloc="s3.amazonaws.com",
                    path=f"/static.nugs.net{parsed.path}",
                ).geturl()
            # Catalog image paths are relative to the service CDN bucket.
            if path.startswith("/") and not path.startswith("//"):
                path = path.lstrip("/")
            image_url = urljoin(IMAGE_BASE, path)
            parsed = urlsplit(image_url)
            if parsed.scheme in ("http", "https") and parsed.hostname:
                return image_url
        except ValueError:
            return None
        return None

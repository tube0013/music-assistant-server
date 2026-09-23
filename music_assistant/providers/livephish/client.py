"""LivePhish APIs with server-side authentication and entitled playback."""

from __future__ import annotations

import asyncio
import hashlib
import time
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit

import aiohttp

# LivePhish Android app public client ID; no Music Assistant registration is available.
CLIENT_ID = "Fujeij8d764ydxcnh4676scsr7f4"
SIGNING_PREFIX = "jdfirj8475jf_"
USER_AGENT = "MusicAssistant/LivePhish"
BASE = "https://streamapi.livephish.com/"
IDENTITY = "https://id.livephish.com/connect/token"
MAX_CACHED_ALBUMS = 128


class LivePhishError(Exception):
    """An API failure with a message safe to share."""


class LivePhishAuthError(LivePhishError):
    """Credentials or subscription access require user attention."""


class LivePhishClient:
    """Keep credentials and subscription state in memory; never store audio."""

    def __init__(self, session: aiohttp.ClientSession, username: str, password: str) -> None:
        """Initialize account state and request pacing."""
        self.session = session
        self.username = username
        self.password = password
        self._lossless_available: bool | None = None
        # Cache raw API responses here so browse, search, and sync share them without
        # coupling the transport to MA. Account-scoped responses expire in memory;
        # these bounded caches intentionally do not persist across provider reloads.
        self._stash_lock = asyncio.Lock()
        self._stash_cache: dict[str, tuple[float, dict[str, Any]]] = {}
        self._access_token = ""
        self._refresh_token = ""
        self._token_expires = 0.0
        self._catalog_lock = asyncio.Lock()
        self._catalog_cache: dict[str, tuple[float, dict[str, Any]]] = {}
        self._auth_lock = asyncio.Lock()
        self._request_lock = asyncio.Lock()
        self._next_request = 0.0
        self._expires = 0.0
        self._subscription: dict[str, Any] = {}
        self._albums: dict[str, tuple[float, dict[str, Any]]] = {}
        self._albums_lock = asyncio.Lock()

    async def authenticate(self) -> None:
        """Refresh account access using modern subscription and user APIs."""
        async with self._auth_lock:
            if time.monotonic() < self._expires:
                return
            self._subscription = {}
            self._lossless_available = None
            self._expires = 0.0
            if not self._access_token or time.monotonic() >= self._token_expires:
                await self._authenticate_token()
            headers = {"Authorization": f"Bearer {self._access_token}"}
            subscriber = await self._request(
                "GET",
                "https://subscriptions.livephish.com/api/v1/me/subscriptions",
                "subscription",
                headers=headers,
            )
            if subscriber.get("isContentAccessible") is not True:
                raise LivePhishAuthError(
                    "subscription: account does not report streaming entitlement"
                )
            plan = subscriber.get("plan")
            if not isinstance(plan, dict) or not plan.get("id"):
                promo = subscriber.get("promo")
                plan = promo.get("plan") if isinstance(promo, dict) else None
            if not isinstance(plan, dict) or not plan.get("id"):
                raise LivePhishError("subscription: missing access plan")
            user = await self._request(
                "GET", "https://stash.livephish.com/api/v1/stash", "user", headers=headers
            )
            try:
                start = datetime.strptime(subscriber["startedAt"], "%m/%d/%Y %H:%M:%S").replace(
                    tzinfo=UTC
                )
                end = datetime.strptime(subscriber["endsAt"], "%m/%d/%Y %H:%M:%S").replace(
                    tzinfo=UTC
                )
            except KeyError, TypeError, ValueError:
                raise LivePhishError("subscription: invalid access dates") from None
            sub = {
                "subscriptionID": subscriber.get("legacySubscriptionId"),
                "subCostplanIDAccessList": plan["id"],
                "userID": user.get("userId"),
                "startDateStamp": int(start.timestamp()),
                "endDateStamp": int(end.timestamp()),
            }
            if any(value in (None, "", 0, "0") for value in sub.values()):
                raise LivePhishError("subscription: missing required access fields")
            self._subscription = sub
            self._lossless_available = str(plan.get("serviceLevel", "")).casefold() == "highquality"
            self._expires = min(time.monotonic() + 900, self._token_expires)

    async def favorite_albums(self) -> list[dict[str, Any]]:
        """Read saved releases without importing the service catalog."""
        return await self._stash_items("releases/favorite")

    async def playlists(self) -> list[dict[str, Any]]:
        """Read all saved playlists from the authenticated account."""
        return await self._stash_items("playlists")

    async def playlist(self, playlist_id: str) -> dict[str, Any]:
        """Read a saved playlist's metadata."""
        self._validate_playlist_id(playlist_id)
        data = await self._stash(f"playlists/{playlist_id}")
        if str(data.get("id")) != playlist_id or not data.get("name"):
            raise LivePhishError("playlists: invalid playlist metadata")
        return data

    async def playlist_tracks(self, playlist_id: str) -> list[dict[str, Any]]:
        """Read playlist performances in their saved order, including duplicates."""
        self._validate_playlist_id(playlist_id)
        return await self._stash_items(f"playlists/{playlist_id}/playlist-tracks/all")

    async def artists(self) -> list[dict[str, Any]]:
        """Discover artists, including side projects, from the service catalog."""
        data = await self._catalog("catalog.artists")
        response = data.get("Response")
        artists = response.get("artists") if isinstance(response, dict) else None
        if not isinstance(artists, list) or any(
            not isinstance(item, dict) or not item.get("artistID") or not item.get("artistName")
            for item in artists
        ):
            raise LivePhishError("catalog: artist list missing or invalid")
        return artists

    async def releases(
        self,
        artist_id: str = "",
        year: str = "",
        offset: int = 0,
        limit: int = 100,
        albums_only: bool = False,
    ) -> list[dict[str, Any]]:
        """Get a page of releases using the service's one-based offset."""
        if (artist_id and not artist_id.isdecimal()) or (year and not year.isdecimal()):
            raise LivePhishError("catalog: artist and year must contain digits only")
        if offset < 0 or not 1 <= limit <= 100:
            raise LivePhishError("catalog: invalid page")
        params: dict[str, str | int] = {
            "limit": limit,
            "startOffset": offset + 1,
            "availtype": 1,
            "sortBy": "albumTitle" if albums_only else "performanceDate",
            "sortType": "asc" if albums_only else "desc",
        }
        if artist_id:
            params["artistList"] = artist_id
        if year:
            params["showYears"] = year
        data = await self._catalog("catalog.containersAll", **params)
        response = data.get("Response")
        items = response.get("containers") if isinstance(response, dict) else None
        if not isinstance(items, list) or any(
            not isinstance(item, dict) or not item.get("containerID") for item in items
        ):
            raise LivePhishError("catalog: release list missing or invalid")
        return items

    async def artist_releases(
        self, artist_id: str, *, albums_only: bool = False
    ) -> list[dict[str, Any]]:
        """Get all artist releases without trusting the legacy API's page-local total."""
        result: dict[str, dict[str, Any]] = {}
        # The legacy API uses sortBy to select shows or albums, not just their order.
        for album_catalog in (True,) if albums_only else (False, True):
            offset = 0
            seen: set[str] = set()
            while True:
                items = await self.releases(
                    artist_id=artist_id, offset=offset, albums_only=album_catalog
                )
                ids = {str(item["containerID"]) for item in items}
                if items and not ids - seen:
                    raise LivePhishError("catalog: release pagination did not advance")
                seen.update(ids)
                result.update((str(item["containerID"]), item) for item in items)
                if len(items) < 100:
                    break
                offset += len(items)
        return list(result.values())

    async def search(self, query: str) -> list[dict[str, Any]]:
        """Flatten the service's grouped show and song search results."""
        if not query.strip():
            return []
        data = await self._catalog("catalog.search", searchStr=query.strip())
        response = data.get("Response")
        if not isinstance(response, dict) or response.get("searchError") not in (None, 0, "0"):
            raise LivePhishError("catalog: search failed")
        groups = response.get("catalogSearchTypeContainers")
        if not isinstance(groups, list):
            raise LivePhishError("catalog: search results missing")
        result = []
        try:
            for group in groups:
                for section in group["catalogSearchContainers"]:
                    for raw_item in section["catalogSearchResultItems"]:
                        if not isinstance(raw_item, dict) or not raw_item.get("containerID"):
                            raise ValueError
                        item = dict(raw_item)
                        if (
                            item.get("trackID")
                            and not item.get("songTitle")
                            and str(section.get("matchType", group.get("matchType"))) == "2"
                            and isinstance(section.get("matchedStr"), str)
                        ):
                            item["songTitle"] = section["matchedStr"]
                        result.append(item)
        except KeyError, TypeError, ValueError:
            raise LivePhishError("catalog: invalid search results") from None
        return result

    async def album(self, album_id: str) -> dict[str, Any]:
        """Get a show's metadata, cached for an hour."""
        if not album_id.isdecimal():
            raise LivePhishError("catalog: show ID must contain digits only")
        cached = self._albums.get(album_id)
        if cached and cached[0] > time.monotonic():
            return cached[1]
        async with self._albums_lock:
            cached = self._albums.get(album_id)
            if cached and cached[0] > time.monotonic():
                return cached[1]
            data = await self._request(
                "GET",
                BASE + "api.aspx",
                "catalog",
                params={
                    "method": "catalog.container",
                    "containerID": album_id,
                    "vdisp": "1",
                },
            )
            album = data.get("Response")
            if not isinstance(album, dict) or str(album.get("containerID")) != album_id:
                raise LivePhishError("catalog: show not found or response schema changed")
            if not isinstance(album.get("tracks"), list):
                raise LivePhishError("catalog: track list missing")
            if album_id not in self._albums and len(self._albums) >= MAX_CACHED_ALBUMS:
                self._albums.pop(next(iter(self._albums)))
            self._albums[album_id] = (time.monotonic() + 3600, album)
            return album

    async def stream(self, track_id: str, quality: str = "aac") -> str:
        """Resolve fresh audio at a quality permitted by the authenticated subscription."""
        if quality not in ("auto", "flac", "alac", "aac"):
            raise LivePhishError("stream: invalid audio quality")
        await self.authenticate()
        if quality != "aac":
            lossless = self._lossless_available is True
            if quality == "auto":
                quality = "flac" if lossless else "aac"
            elif not lossless:
                raise LivePhishError(
                    "stream: lossless audio requires a LivePhish HighQuality subscription"
                )
        sub = self._subscription
        stamp = f"{int(time.time() * 1000) + 60000}.000000"
        signature = hashlib.md5(
            (SIGNING_PREFIX + stamp).encode(), usedforsecurity=False
        ).hexdigest()
        data = await self._request(
            "GET",
            BASE + "bigriver/subplayer.aspx",
            "stream",
            params={
                "trackId": track_id,
                "app": "1",
                "HLS": "1",
                "orgn": "websdk",
                "method": "subPlayer",
                "platformID": {"aac": "4", "alac": "2", "flac": "3"}[quality],
                "subscriptionID": str(sub["subscriptionID"]),
                "subCostplanIDAccessList": str(sub["subCostplanIDAccessList"]),
                "nn_userID": str(sub["userID"]),
                "startDateStamp": str(sub["startDateStamp"]),
                "endDateStamp": str(sub["endDateStamp"]),
                "tk": signature,
                "lxp": stamp,
            },
        )
        link = data.get("streamLink")
        if not isinstance(link, str) or not link.strip():
            raise LivePhishError("stream: LivePhish returned no audio URL")
        try:
            parsed = urlsplit(link)
            valid = (
                parsed.scheme in ("http", "https")
                and bool(parsed.hostname)
                and parsed.username is None
                and parsed.password is None
            )
        except ValueError:
            valid = False
        if not valid:
            raise LivePhishError("stream: LivePhish returned an invalid HTTP(S) audio URL")
        return link

    async def _authenticate_token(self) -> None:
        data = {"client_id": CLIENT_ID}
        token: dict[str, Any] | None = None
        if self._refresh_token:
            try:
                token = await self._request(
                    "POST",
                    IDENTITY,
                    "refresh",
                    data={
                        **data,
                        "grant_type": "refresh_token",
                        "refresh_token": self._refresh_token,
                    },
                )
            except LivePhishAuthError:
                self._refresh_token = ""
        if token is None:
            token = await self._request(
                "POST",
                IDENTITY,
                "login",
                data={
                    **data,
                    "grant_type": "password",
                    "scope": "offline_access nugsnet:api nugsnet:legacyapi",
                    "username": self.username,
                    "password": self.password,
                },
            )
        access = token.get("access_token")
        if not isinstance(access, str) or not access:
            raise LivePhishError("login: no access token")
        try:
            ttl = float(token["expires_in"])
            if not 0 < ttl < float("inf"):
                raise ValueError
        except KeyError, TypeError, ValueError:
            raise LivePhishError("login: invalid token expiry") from None
        self._access_token = access
        refresh = token.get("refresh_token")
        if isinstance(refresh, str) and refresh:
            self._refresh_token = refresh
        self._token_expires = time.monotonic() + ttl - min(60, ttl / 10)

    def _validate_playlist_id(self, playlist_id: str) -> None:
        if not playlist_id or not all(char.isalnum() or char in "-_" for char in playlist_id):
            raise LivePhishError("playlists: invalid playlist ID")

    async def _stash_items(self, endpoint: str) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        seen_pages: set[tuple[str, ...]] = set()
        while True:
            data = await self._stash(endpoint, limit=100, offset=len(result))
            items = data.get("items")
            if not isinstance(items, list) or any(
                not isinstance(item, dict) or not item.get("id") for item in items
            ):
                raise LivePhishError("playlists: invalid item list")
            if not items:
                return result
            page_ids = tuple(str(item["id"]) for item in items)
            if page_ids in seen_pages:
                raise LivePhishError("playlists: pagination did not advance")
            seen_pages.add(page_ids)
            result.extend(items)
            try:
                total = int(data["total"]) if "total" in data else None
            except ValueError, TypeError:
                raise LivePhishError("playlists: invalid item count") from None
            if total is not None and len(result) >= total:
                return result
            if total is None and len(items) < 100:
                return result

    async def _stash(self, endpoint: str, **params: int) -> dict[str, Any]:
        async with self._stash_lock:
            await self.authenticate()
            key = repr((endpoint, sorted(params.items())))
            cached = self._stash_cache.get(key)
            if cached and cached[0] > time.monotonic():
                return cached[1]
            data = await self._request(
                "GET",
                f"https://stash.livephish.com/api/v1/me/{endpoint}",
                "playlists",
                headers={"Authorization": f"Bearer {self._access_token}"},
                params=params,
            )
            if len(self._stash_cache) >= 128:
                self._stash_cache.pop(next(iter(self._stash_cache)))
            self._stash_cache[key] = (time.monotonic() + 60, data)
            return data

    async def _catalog(self, method: str, **params: str | int) -> dict[str, Any]:
        # Coalesce concurrent cache misses before the shared request throttle.
        key = repr((method, sorted(params.items())))
        async with self._catalog_lock:
            cached = self._catalog_cache.get(key)
            if cached and cached[0] > time.monotonic():
                return cached[1]
            data = await self._request(
                "GET", BASE + "api.aspx", "catalog", params={"method": method, **params}
            )
            if data.get("responseAvailabilityCode") not in (None, 0, "0"):
                raise LivePhishError("catalog: service reports catalog unavailable")
            if not isinstance(data.get("Response"), (dict, list)):
                raise LivePhishError("catalog: invalid response")
            if len(self._catalog_cache) >= 128:
                self._catalog_cache.pop(next(iter(self._catalog_cache)))
            self._catalog_cache[key] = (time.monotonic() + 3600, data)
            return data

    async def _request(self, method: str, url: str, stage: str, **kwargs: Any) -> dict[str, Any]:
        headers = {"User-Agent": USER_AGENT, **kwargs.pop("headers", {})}
        # Serial requests plus a one-second minimum interval also protect standalone probes.
        async with self._request_lock:
            await asyncio.sleep(max(0, self._next_request - time.monotonic()))
            try:
                async with self.session.request(
                    method,
                    url,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30),
                    allow_redirects=False,
                    **kwargs,
                ) as response:
                    self._next_request = time.monotonic() + 1
                    if response.status == 429:
                        try:
                            delay = min(3600, max(30, int(response.headers.get("Retry-After", 60))))
                        except ValueError:
                            delay = 60
                        self._next_request = time.monotonic() + delay
                        raise LivePhishError(f"{stage}: rate limited; wait {delay} seconds")
                    if response.status in (400, 401) and stage in ("login", "refresh"):
                        failure = await response.json(content_type=None)
                        if isinstance(failure, dict) and failure.get("error") == "invalid_grant":
                            raise LivePhishAuthError(f"{stage}: credentials expired or rejected")
                    if response.status == 401:
                        self._token_expires = 0.0
                        self._expires = 0.0
                    if response.status != 200:
                        raise LivePhishError(f"{stage}: HTTP {response.status}")
                    data = await response.json(content_type=None)
                    if not isinstance(data, dict):
                        raise LivePhishError(f"{stage}: unexpected response shape")
                    return data
            except aiohttp.ClientError, TimeoutError, ValueError:
                # aiohttp errors can contain credential-bearing request URLs.
                raise LivePhishError(f"{stage}: connection or JSON response failure") from None
            finally:
                self._next_request = max(self._next_request, time.monotonic() + 1)

"""Modern authentication must preserve entitlement and keep credentials out of URLs."""

import hashlib
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from music_assistant.providers.livephish.client import (
    IDENTITY,
    USER_AGENT,
    LivePhishAuthError,
    LivePhishClient,
    LivePhishError,
)

SUB = {
    "isContentAccessible": True,
    "legacySubscriptionId": "service-subscription",
    "plan": {"id": "service-plan", "serviceLevel": "HighQuality"},
    "startedAt": "01/01/2026 00:00:00",
    "endsAt": "01/01/2027 00:00:00",
}
TOKEN = {"access_token": "access", "refresh_token": "refresh", "expires_in": 3600}


@pytest.fixture
def client() -> LivePhishClient:
    """Use synthetic credentials and no real network connection."""
    return LivePhishClient(MagicMock(), "test-user", "test-password")


async def test_modern_login_and_entitlement(client: LivePhishClient) -> None:
    """Send credentials only in the identity POST and use authenticated modern APIs."""
    with patch.object(client, "_request", new_callable=AsyncMock) as request:
        request.side_effect = [TOKEN, SUB, {"userId": "service-user"}]
        await client.authenticate()
        await client.authenticate()
        calls = request.call_args_list
        assert len(calls) == 3
        assert calls[0].args == ("POST", IDENTITY, "login")
        assert calls[0].kwargs["data"]["password"] == "test-password"
        assert "params" not in calls[0].kwargs
        assert calls[1].args[1] == "https://subscriptions.livephish.com/api/v1/me/subscriptions"
        assert calls[2].args[1] == "https://stash.livephish.com/api/v1/stash"
        assert calls[1].kwargs["headers"] == {"Authorization": "Bearer access"}
        assert client._subscription == {
            "subscriptionID": "service-subscription",
            "subCostplanIDAccessList": "service-plan",
            "userID": "service-user",
            "startDateStamp": 1767225600,
            "endDateStamp": 1798761600,
        }
        assert client._lossless_available is True
        assert USER_AGENT == "MusicAssistant/LivePhish"


async def test_token_refresh_rotates_without_password(client: LivePhishClient) -> None:
    """Refresh expired tokens without repeating password authentication."""
    client._refresh_token = "old-refresh"
    with patch.object(client, "_request", new_callable=AsyncMock) as request:
        request.side_effect = [TOKEN, SUB, {"userId": "service-user"}]
        await client.authenticate()
        data = request.call_args_list[0].kwargs["data"]
        assert data["grant_type"] == "refresh_token"
        assert data["refresh_token"] == "old-refresh"
        assert "password" not in data
        assert client._refresh_token == "refresh"


async def test_refresh_rejection_allows_password_relogin(client: LivePhishClient) -> None:
    """An explicitly rejected refresh token may use the configured login once."""
    client._refresh_token = "rejected-refresh"
    with patch.object(client, "_request", new_callable=AsyncMock) as request:
        request.side_effect = [LivePhishAuthError("refresh: rejected"), TOKEN, SUB, {"userId": "u"}]
        await client.authenticate()
        assert request.call_args_list[1].kwargs["data"]["grant_type"] == "password"


async def test_refresh_outage_never_falls_back_to_password(client: LivePhishClient) -> None:
    """Temporary errors retain refresh credentials and do not trigger another login."""
    client._refresh_token = "retained-refresh"
    with patch.object(client, "_request", new_callable=AsyncMock) as request:
        request.side_effect = LivePhishError("refresh: HTTP 503")
        with pytest.raises(LivePhishError, match="503"):
            await client.authenticate()
        request.assert_awaited_once()
        assert client._refresh_token == "retained-refresh"
        assert not client._subscription


async def test_entitlement_refresh_reuses_valid_token(client: LivePhishClient) -> None:
    """The periodic entitlement check need not authenticate again."""
    client._access_token = "valid-access"
    client._token_expires = time.monotonic() + 3600
    client._lossless_available = True
    with patch.object(client, "_request", new_callable=AsyncMock) as request:
        request.side_effect = [
            {**SUB, "plan": {"id": "p", "serviceLevel": "Standard"}},
            {"userId": "u"},
        ]
        await client.authenticate()
        assert request.await_count == 2
        assert all(call.args[0] == "GET" for call in request.call_args_list)
        assert client._lossless_available is False


@pytest.mark.parametrize("accessible", [False, None, "true", 1])
async def test_inaccessible_account_fails_closed(
    client: LivePhishClient, accessible: object
) -> None:
    """Only an explicit service entitlement permits streaming."""
    with patch.object(client, "_request", new_callable=AsyncMock) as request:
        request.side_effect = [TOKEN, {**SUB, "isContentAccessible": accessible}]
        with pytest.raises(LivePhishAuthError, match="entitlement"):
            await client.stream("29990")
        assert request.await_count == 2
        assert not client._subscription


@pytest.mark.parametrize(
    "change",
    [
        {"legacySubscriptionId": None},
        {"plan": None},
        {"plan": []},
        {"promo": "bad", "plan": None},
        {"startedAt": None},
        {"endsAt": "invalid"},
    ],
)
async def test_malformed_entitlement(client: LivePhishClient, change: dict[str, object]) -> None:
    """Malformed account data must fail safely before requesting any audio."""
    with patch.object(client, "_request", new_callable=AsyncMock) as request:
        request.side_effect = [TOKEN, {**SUB, **change}, {"userId": "u"}]
        with pytest.raises(LivePhishError):
            await client.stream("29990")
        assert not client._subscription


async def test_promo_plan(client: LivePhishClient) -> None:
    """A service-issued promo plan retains its own access ID and quality."""
    with patch.object(client, "_request", new_callable=AsyncMock) as request:
        request.side_effect = [
            TOKEN,
            {**SUB, "plan": None, "promo": {"plan": SUB["plan"]}},
            {"userId": "u"},
        ]
        await client.authenticate()
        assert client._subscription["subCostplanIDAccessList"] == "service-plan"
        assert client._lossless_available is True


@pytest.mark.parametrize("ttl", [0, -1, None, "invalid", float("inf"), float("nan")])
async def test_invalid_token_expiry(client: LivePhishClient, ttl: object) -> None:
    """Reject unusable expiry without publishing token or entitlement state."""
    with patch.object(client, "_request", new_callable=AsyncMock) as request:
        request.return_value = {**TOKEN, "expires_in": ttl}
        with pytest.raises(LivePhishError, match="expiry"):
            await client.authenticate()
        assert not client._access_token


async def test_short_token_lifetime(client: LivePhishClient) -> None:
    """Short-lived tokens retain a proportional margin instead of a one-second loop."""
    with (
        patch.object(client, "_request", new_callable=AsyncMock) as request,
        patch("music_assistant.providers.livephish.client.time.monotonic", return_value=100),
    ):
        request.side_effect = [{**TOKEN, "expires_in": 30}, SUB, {"userId": "u"}]
        await client.authenticate()
        assert client._token_expires == 127
        assert client._expires == 127


async def test_web_stream_request(client: LivePhishClient) -> None:
    """Use observed web signing and preserve the URL returned by the service."""
    with (
        patch.object(client, "_request", new_callable=AsyncMock) as request,
        patch("music_assistant.providers.livephish.client.time.time", return_value=1700000000),
    ):
        link = "http://audio.example.test/track?sig=a%2Bb"
        request.side_effect = [TOKEN, SUB, {"userId": "u"}, {"streamLink": link}]
        assert await client.stream("29990", "flac") == link
        call = request.call_args
        params = call.kwargs["params"]
        assert params["orgn"] == "websdk"
        assert params["method"] == "subPlayer"
        assert params["platformID"] == "3"
        assert params["lxp"] == "1700000060000.000000"
        assert params["tk"] == hashlib.md5(b"jdfirj8475jf_1700000060000.000000").hexdigest()
        assert "headers" not in call.kwargs

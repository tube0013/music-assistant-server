"""Regression coverage for service-issued LivePhish audio URLs."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from music_assistant.providers.livephish.client import LivePhishClient, LivePhishError


@pytest.fixture
def client() -> LivePhishClient:
    """Create a client with synthetic entitlement and no network access."""
    client = LivePhishClient(MagicMock(), "test-user", "test-password")
    client._subscription = {
        "subscriptionID": "test-subscription",
        "subCostplanIDAccessList": "test-plan",
        "userID": "test-user",
        "startDateStamp": "test-start",
        "endDateStamp": "test-end",
    }
    return client


@pytest.mark.parametrize("scheme", ["http", "https"])
async def test_stream_preserves_signed_url(client: LivePhishClient, scheme: str) -> None:
    """Accept both audio transports without normalizing signature-sensitive bytes."""
    link = f"{scheme}://audio.example.test/Axilla%2fa.m4a?sig=a%2Bb%2Fc%3D&x=1&x=2"
    with (
        patch.object(client, "authenticate", new_callable=AsyncMock) as authenticate,
        patch.object(client, "_request", new_callable=AsyncMock) as request,
    ):
        request.return_value = {"streamLink": link}
        assert await client.stream("29990") == link
        authenticate.assert_awaited_once_with()
        assert request.call_args.args[1].startswith("https://")
        assert request.call_args.kwargs["params"]["trackId"] == "29990"
        assert request.call_args.kwargs["params"]["subscriptionID"] == "test-subscription"


@pytest.mark.parametrize("link", [None, "", "   ", 123, {}, []])
async def test_stream_missing_url(client: LivePhishClient, link: object) -> None:
    """Handle absent or non-string stream links with a safe error."""
    with (
        patch.object(client, "authenticate", new_callable=AsyncMock),
        patch.object(client, "_request", new_callable=AsyncMock) as request,
    ):
        request.return_value = {"streamLink": link}
        with pytest.raises(LivePhishError, match=r"^stream: LivePhish returned no audio URL$"):
            await client.stream("29990")


@pytest.mark.parametrize(
    "link",
    [
        "ftp://audio.example.test/track?sig=secret",
        "file:///track?sig=secret",
        "//audio.example.test/track?sig=secret",
        "https:///track?sig=secret",
        "http://?sig=secret",
        "https://user:secret@audio.example.test/track",
        "http://user@audio.example.test/track?sig=secret",
        "https://[invalid/track?sig=secret",
    ],
)
async def test_stream_invalid_url(client: LivePhishClient, link: str) -> None:
    """Reject malformed or credential-bearing links without disclosing their contents."""
    with (
        patch.object(client, "authenticate", new_callable=AsyncMock),
        patch.object(client, "_request", new_callable=AsyncMock) as request,
    ):
        request.return_value = {"streamLink": link}
        with pytest.raises(LivePhishError) as error:
            await client.stream("29990")
        assert str(error.value) == "stream: LivePhish returned an invalid HTTP(S) audio URL"


async def test_stream_requires_entitlement(client: LivePhishClient) -> None:
    """Never request audio when account authentication or entitlement fails."""
    with (
        patch.object(client, "authenticate", new_callable=AsyncMock) as authenticate,
        patch.object(client, "_request", new_callable=AsyncMock) as request,
    ):
        authenticate.side_effect = LivePhishError("subscription: no streaming entitlement")
        with pytest.raises(LivePhishError, match="no streaming entitlement"):
            await client.stream("29990")
        request.assert_not_awaited()

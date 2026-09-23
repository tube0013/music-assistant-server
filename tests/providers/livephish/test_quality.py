"""Quality selection must respect account entitlement and preserve signed URLs."""

from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from music_assistant_models.enums import ContentType, MediaType

from music_assistant.providers.livephish import LivePhishProvider
from music_assistant.providers.livephish.client import LivePhishClient, LivePhishError
from tests.providers.livephish.fixtures import RELEASES

SUBSCRIPTION = {
    "subscriptionID": "issued-sub",
    "subCostplanIDAccessList": "issued-plan",
    "userID": "issued-user",
    "startDateStamp": "issued-start",
    "endDateStamp": "issued-end",
}
LINK = "http://audio.example.test/track?sig=a%2Bb%2Fc%3D&x=1&x=2"


@pytest.fixture
def client() -> LivePhishClient:
    """Create an isolated client with synthetic account state."""
    client = LivePhishClient(MagicMock(), "test-user", "test-password")
    client._subscription = dict(SUBSCRIPTION)
    client._access_token = "synthetic-access-token"
    client._lossless_available = True
    return client


@pytest.mark.parametrize(
    ("quality", "platform"), [("auto", "3"), ("flac", "3"), ("alac", "2"), ("aac", "4")]
)
async def test_quality_format_and_issued_entitlement(
    client: LivePhishClient,
    quality: str,
    platform: str,
) -> None:
    """Map formats while keeping account-issued access fields and URL bytes unchanged."""
    with (
        patch.object(client, "authenticate", new_callable=AsyncMock),
        patch.object(client, "_request", new_callable=AsyncMock) as request,
    ):
        request.return_value = {"streamLink": LINK}
        assert await client.stream("29990", quality) == LINK
        params = request.call_args.kwargs["params"]
        assert params["platformID"] == platform
        assert params["subCostplanIDAccessList"] == SUBSCRIPTION["subCostplanIDAccessList"]
        assert client._subscription == SUBSCRIPTION
        request.assert_awaited_once()


@pytest.mark.parametrize("quality", ["flac", "alac"])
async def test_standard_subscription_cannot_request_lossless(
    client: LivePhishClient, quality: str
) -> None:
    """Explicit lossless requests fail before audio resolution without HighQuality access."""
    with (
        patch.object(client, "authenticate", new_callable=AsyncMock),
        patch.object(client, "_request", new_callable=AsyncMock) as request,
    ):
        client._lossless_available = False
        with pytest.raises(LivePhishError, match="requires a LivePhish HighQuality subscription"):
            await client.stream("29990", quality)
        request.assert_not_awaited()


@pytest.mark.parametrize("entitlement", [False, None])
async def test_automatic_keeps_aac_available(
    client: LivePhishClient, entitlement: bool | None
) -> None:
    """Only confirmed lossless entitlement selects FLAC automatically."""
    client._lossless_available = entitlement
    with (
        patch.object(client, "authenticate", new_callable=AsyncMock),
        patch.object(client, "_request", new_callable=AsyncMock) as request,
    ):
        request.return_value = {"streamLink": LINK}
        assert await client.stream("29990", "auto") == LINK
        assert request.call_args.kwargs["params"]["platformID"] == "4"
        request.assert_awaited_once()


async def test_auth_refresh_invalidates_quality_entitlement(client: LivePhishClient) -> None:
    """Reauthentication cannot retain lossless access from an older subscription state."""
    client._lossless_available = True
    with patch.object(client, "_request", new_callable=AsyncMock) as request:
        request.side_effect = [
            {"access_token": "new-token", "expires_in": 3600},
            {
                "isContentAccessible": True,
                "legacySubscriptionId": "issued-sub",
                "plan": {"id": "issued-plan", "serviceLevel": "Standard"},
                "startedAt": "07/02/2026 18:07:22",
                "endsAt": "07/02/2027 18:07:22",
            },
            {"userId": "issued-user"},
        ]
        await client.authenticate()
        assert client._lossless_available is False


async def test_invalid_quality_does_not_request_audio(client: LivePhishClient) -> None:
    """Reject unknown quality values without making service requests."""
    with patch.object(client, "authenticate", new_callable=AsyncMock) as authenticate:
        with pytest.raises(LivePhishError, match="invalid audio quality"):
            await client.stream("29990", "unlimited")
        authenticate.assert_not_awaited()


@pytest.mark.parametrize("quality", [None, "auto", "flac", "alac", "aac"])
async def test_provider_forwards_quality_and_probes_actual_codec(
    provider: LivePhishProvider,
    quality: str | None,
) -> None:
    """Wire configuration into playback without advertising an unverified codec."""
    client = cast("MagicMock", provider._client)
    client.album.return_value = {**RELEASES[0], "tracks": [{"trackID": 29990, "songTitle": "Test"}]}
    client.stream.return_value = LINK
    provider.config.get_value = MagicMock(return_value=quality)  # type: ignore[method-assign]
    details = await provider.get_stream_details("1773:29990", MediaType.TRACK)
    client.stream.assert_awaited_once_with("29990", quality=quality or "auto")
    assert details.path == LINK
    assert details.can_seek is True
    assert details.allow_seek is True
    assert details.audio_format.content_type == ContentType.UNKNOWN

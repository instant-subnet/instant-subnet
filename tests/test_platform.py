import httpx
import pytest

from instant_validator.config import Settings
from instant_validator.platform import PlatformClient, PlatformError

SIGNER = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"


def settings():
    return Settings.from_env(
        {
            "INSTANT_PLATFORM_SIGNER": SIGNER,
            "INSTANT_PLATFORM_REPORT_URL": "https://platform.example/reports/latest",
        },
        load_env_file=False,
    )


def test_fetches_and_validates_one_latest_report(report_raw):
    def handler(request):
        assert request.url == "https://platform.example/reports/latest"
        return httpx.Response(200, content=report_raw)

    http = httpx.Client(transport=httpx.MockTransport(handler))
    report = PlatformClient(settings(), http=http).fetch_latest(now_ms=1_786_708_810_000)

    assert report.netuid == 46
    http.close()


def test_http_failure_is_not_treated_as_an_empty_report():
    http = httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(503, text="unavailable"))
    )
    with pytest.raises(PlatformError, match="request failed"):
        PlatformClient(settings(), http=http).fetch_latest()
    http.close()

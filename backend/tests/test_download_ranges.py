from __future__ import annotations

import sys
import urllib.error
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import download_ranges  # noqa: E402


class _Response:
    headers = {"Content-Length": "1234", "Accept-Ranges": "bytes"}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def test_remote_size_retries_transient_connection_errors(monkeypatch):
    calls = 0
    sleeps: list[int] = []

    def urlopen(_request, timeout):
        nonlocal calls
        assert timeout == 7
        calls += 1
        if calls < 3:
            raise urllib.error.URLError("temporary timeout")
        return _Response()

    monkeypatch.setattr(download_ranges.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(download_ranges.time, "sleep", sleeps.append)

    assert download_ranges.remote_size("https://example.invalid/model", attempts=3, timeout=7) == 1234
    assert calls == 3
    assert sleeps == [2, 4]


def test_remote_size_does_not_retry_missing_range_support(monkeypatch):
    class NoRanges(_Response):
        headers = {"Content-Length": "1234"}

    monkeypatch.setattr(download_ranges.urllib.request, "urlopen", lambda *_args, **_kwargs: NoRanges())

    try:
        download_ranges.remote_size("https://example.invalid/model", attempts=5, timeout=1)
    except RuntimeError as exc:
        assert "does not advertise byte ranges" in str(exc)
    else:
        raise AssertionError("missing Accept-Ranges must fail")

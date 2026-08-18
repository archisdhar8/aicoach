import pytest

from mlb_winners.net import ensure_host_resolves


def test_ensure_host_resolves_localhost():
    assert ensure_host_resolves("http://localhost") == "localhost"


def test_ensure_host_resolves_invalid_host_raises():
    with pytest.raises(RuntimeError):
        ensure_host_resolves("https://definitely-not-a-real-hostname.invalid")

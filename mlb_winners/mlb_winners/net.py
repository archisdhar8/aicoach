from __future__ import annotations

import socket
from urllib.parse import urlparse


def ensure_host_resolves(url_or_host: str) -> str:
    """Fail fast when DNS/network is unavailable.

    Returns the resolved hostname that was checked.
    """
    parsed = urlparse(url_or_host)
    host = parsed.hostname or url_or_host
    if not host:
        raise ValueError("url_or_host must include a hostname")
    try:
        socket.getaddrinfo(host, None)
    except OSError as exc:
        raise RuntimeError(f"Cannot resolve host '{host}'. Network/DNS may be unavailable.") from exc
    return host

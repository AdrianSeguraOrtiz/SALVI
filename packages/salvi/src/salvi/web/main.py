"""Local web-server launcher used by ``salvi gui``."""

from __future__ import annotations

import ipaddress
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

from salvi.exceptions import ConfigurationError, OptionalDependencyError


def _is_loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def launch_web_gui(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
    data_directory: Path | None = None,
    max_upload_mib: int = 2048,
) -> int:
    if not _is_loopback(host):
        raise ConfigurationError(
            "SALVI GUI only accepts loopback interfaces because it has no authentication"
        )
    try:
        import uvicorn

        from salvi.web.app import create_app
    except ModuleNotFoundError as error:
        raise OptionalDependencyError(
            "the web GUI dependencies are unavailable; reinstall the complete 'salvi' package"
        ) from error

    app = create_app(
        data_directory=data_directory,
        max_upload_mib=max_upload_mib,
    )
    url_host = "127.0.0.1" if host in {"0:0:0:0:0:0:0:1", "::1"} else host
    url = f"http://{url_host}:{port}"
    print(f"SALVI web interface: {url}")
    print(
        "When running over VS Code Remote SSH, forward this port in the Ports panel "
        "and open the forwarded local address."
    )

    if open_browser:
        threading.Thread(
            target=_open_when_ready,
            args=(f"{url}/api/v1/health", url),
            name="salvi-browser",
            daemon=True,
        ).start()
    uvicorn.run(app, host=host, port=port, log_level="info")
    return 0


def _open_when_ready(health_url: str, browser_url: str) -> None:
    for _ in range(80):
        try:
            with urllib.request.urlopen(health_url, timeout=0.25):
                webbrowser.open(browser_url)
                return
        except OSError:
            time.sleep(0.1)


__all__ = ["launch_web_gui"]

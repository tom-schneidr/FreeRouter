from __future__ import annotations

import argparse
import os
from pathlib import Path

import uvicorn

from app.desktop_runtime import DEFAULT_HOST, DEFAULT_PORT
from app.desktop_settings import launcher_host_port


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the FreeRouter backend sidecar.")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    args = parser.parse_args()

    root = os.environ.get("FREEROUTER_DESKTOP_PROJECT_ROOT")
    env_host, env_port = launcher_host_port(Path(root or os.getcwd()))
    host = args.host or os.getenv("GATEWAY_HOST") or env_host or DEFAULT_HOST
    port = args.port or int(os.getenv("GATEWAY_PORT") or env_port or DEFAULT_PORT)

    uvicorn.run("app.main:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()

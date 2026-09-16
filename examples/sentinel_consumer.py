"""Minimal server-side Sentinel preflight, explicit fallback, and receipt example."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, replace
from typing import Any


@dataclass(frozen=True)
class Receipt:
    run_id: str
    profile: str
    provider: str
    route: str
    model: str
    policy_verdict: str
    latency_ms: int
    fallback: str
    fallback_reason: str


def _base_url() -> str:
    return os.getenv("FREEROUTER_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")


def preflight(profile: str, *, tools: bool) -> dict[str, Any]:
    query = urllib.parse.urlencode(
        {
            "profile": profile,
            "chat": "true",
            "json": "true",
            "stream": "true",
            "tools": str(tools).lower(),
        }
    )
    with urllib.request.urlopen(
        f"{_base_url()}/gateway/sentinel/preflight?{query}", timeout=5
    ) as response:
        return json.loads(response.read().decode("utf-8"))


def _receipt(headers: Any) -> Receipt:
    return Receipt(
        run_id=headers.get("X-FreeRouter-Run-Id", ""),
        profile=headers.get("X-FreeRouter-Profile", ""),
        provider=headers.get("X-Gateway-Provider", ""),
        route=headers.get("X-Gateway-Route", ""),
        model=headers.get("X-Gateway-Model", ""),
        policy_verdict=headers.get("X-FreeRouter-Policy-Verdict", ""),
        latency_ms=int(headers.get("X-FreeRouter-Latency-Ms", "0")),
        fallback=headers.get("X-FreeRouter-Fallback", ""),
        fallback_reason=headers.get("X-FreeRouter-Fallback-Reason", ""),
    )


def chat(messages: list[dict[str, str]], *, profile: str, tools: bool) -> tuple[dict, Receipt]:
    health = preflight(profile, tools=tools)
    fallback_model = os.getenv("FREEROUTER_FALLBACK_MODEL", "auto")
    primary_model = profile if health["status"] != "blocked" else fallback_model

    def send(model: str) -> tuple[dict, Receipt]:
        payload = json.dumps({"model": model, "messages": messages}).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        api_key = os.getenv("FREEROUTER_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = urllib.request.Request(
            f"{_base_url()}/chat/completions",
            data=payload,
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=90) as response:
            receipt = _receipt(response.headers)
            return json.loads(response.read().decode("utf-8")), receipt

    if primary_model == fallback_model:
        response, receipt = send(fallback_model)
        return response, replace(
            receipt,
            fallback="consumer-auto",
            fallback_reason=health["reason"],
        )
    try:
        return send(primary_model)
    except urllib.error.HTTPError as error:
        body = json.loads(error.read().decode("utf-8"))
        if body.get("error", {}).get("code") != "no_qualifying_route":
            raise
        response, receipt = send(fallback_model)
        return response, replace(
            receipt,
            fallback="consumer-auto",
            fallback_reason=body["error"]["message"],
        )


if __name__ == "__main__":
    selected = os.getenv("FREEROUTER_MODEL", "safe-study")
    response, receipt = chat(
        [{"role": "user", "content": "Return a two-step plan."}],
        profile=selected,
        tools=selected == "safe-security",
    )
    print(json.dumps({"model": response.get("model"), "receipt": receipt.__dict__}, indent=2))

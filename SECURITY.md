# Security policy

FreeRouter is a local gateway. Treat it as a development tool and review its configuration before
exposing it beyond the host machine.

## Reporting a vulnerability

Please report security issues privately through GitHub's Security Advisories for this repository. If
that channel is unavailable, open a minimal issue asking for a private contact method; do not include
credentials, provider responses, or exploit details in a public issue.

## Scope and safe defaults

- Keep the gateway bound to `127.0.0.1` unless you have added authentication and network controls.
- Keep `.env`, SQLite state, backups, and provider keys out of commits and support bundles.
- Sentinel evidence is local test evidence, not a security certification or a guarantee about an
  upstream provider.
- Review model-catalog changes before enabling newly discovered routes.

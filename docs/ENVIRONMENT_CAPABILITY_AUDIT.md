# Environment Capability Audit

Audit date: 2026-09-28. Performed at the start of Foundation 1.0, before any code was written.
Every value below was observed directly with a command; nothing is inferred from documentation alone.

Classification key:

| Class | Meaning |
|---|---|
| AVAILABLE | Observed working. |
| PARTIALLY_AVAILABLE | Works with a documented restriction. |
| BLOCKED | Observed failing because of the environment (not the architecture). |
| UNKNOWN | Not verifiable from inside this environment. |

Every row also says whether a limitation is **ENVIRONMENT** (a property of this container / its
policy, fixable by configuration, no code change needed) or **ARCHITECTURE** (a design property of
this system).

## Repository

| Item | Observed value |
|---|---|
| Repository root | `/home/user/autonomous-trading-agent` |
| Remote | `https://github.com/youngdizzy/autonomous-trading-agent` |
| Branch | `claude/trading-intelligence-foundation-nix4vy` (also `main`) |
| Starting HEAD | `fff46da2a12321f44311ad86ab6f6afca85a9249` ("Initial commit") |
| Starting working tree | clean; only `README.md` (26 bytes) |
| Legacy TradeTown repo | not present, not accessed, not referenced |

## Runtime and tools

| Capability | Class | Observed |
|---|---|---|
| OS | AVAILABLE | Ubuntu 24.04.4 LTS, Linux 6.18 x86_64, 4 vCPU, 16 GB RAM |
| Python | AVAILABLE | 3.11.15 (`/usr/local/bin/python3`) |
| Python stdlib crypto / numerics | AVAILABLE | `hashlib`, `hmac`, `secrets`, `decimal`, `statistics`, `json`, `sqlite3` (SQLite 3.45.1) |
| Third-party Python packages | PARTIALLY_AVAILABLE | Not preinstalled (no numpy/pandas/pydantic/pytest). `pip install` from PyPI works (pypi.org is on the proxy allow list). |
| pytest | AVAILABLE | installed this session: pytest 9.1.1 |
| Node | AVAILABLE | v22.22.2 (not used by this system) |
| git | AVAILABLE | local commits work; `git ls-remote origin` works |
| uv / poetry / docker binaries | AVAILABLE (unused) | present; deliberately not used (no container infra required) |

Design consequence (ARCHITECTURE decision): the runtime depends **only on the Python standard
library**. pytest is a test-only dependency. This keeps the system portable to any Python 3.11+
environment, including future Claude environments, without package installation.

## Network

Outbound HTTPS passes through an egress proxy that enforces the organization's network policy.

| Destination | Class | Observed | Limitation type |
|---|---|---|---|
| `api.kraken.com` (public OHLC/Time) | BLOCKED | proxy `403 Forbidden` on CONNECT | ENVIRONMENT |
| `api.binance.com`, `data-api.binance.vision` | BLOCKED | proxy 403 | ENVIRONMENT |
| `api.coinbase.com`, `api.exchange.coinbase.com` | BLOCKED | proxy 403 | ENVIRONMENT |
| `paper-api.alpaca.markets` | BLOCKED | proxy 403 | ENVIRONMENT |
| `query1.finance.yahoo.com` | BLOCKED | proxy 403 | ENVIRONMENT |
| `api.coingecko.com` | BLOCKED | proxy 403 | ENVIRONMENT |
| `www.google.com` | BLOCKED | proxy 403 | ENVIRONMENT |
| `pypi.org` | AVAILABLE | HTTP 200 | — |
| `github.com` | AVAILABLE | reachable (HTTP 400 to bare GET; git over proxy works) | — |
| `api.anthropic.com` | AVAILABLE (reachability only) | HTTP 404 on bare GET | — |

Therefore:

```
LIVE_MARKET_DATA      = BLOCKED — ENVIRONMENT CAPABILITY (egress policy denies all tested market-data hosts)
LIVE_CONNECTIVITY     = BLOCKED — ENVIRONMENT CAPABILITY
LIVE_TRADING          = false (ARCHITECTURE: hard-disabled in code, independent of network)
```

This is an environment limitation, not an architectural one. The Kraken adapter is implemented
against the documented public OHLC response shape and contract-tested locally against a
hand-constructed fixture that is labelled MOCK. **It has never received a real Kraken response.**
To lift the block, the environment owner adds `api.kraken.com` to the allowed domains in the cloud
environment's network settings (session title bar → environment menu → Edit → Network access).
No code change is required.

## Claude capabilities in this environment

| Capability | Class | Notes |
|---|---|---|
| File read/write/edit, shell | AVAILABLE | inside the container |
| Claude as reasoning layer (this session) | AVAILABLE | Claude-native mode: the deterministic system emits a context packet, Claude writes a proposal file, the deterministic system validates it. See `docs/ARCHITECTURE.md`. |
| Programmatic Claude API calls from the system | BLOCKED — CREDENTIALS | host reachable, but no API key is provisioned for this project; the session's own credentials must not be reused by the product. Adapter interface exists; HTTP adapter NOT IMPLEMENTED. |
| Sub-agents | AVAILABLE | not needed for Foundation 1.0 |
| MCP: GitHub | AVAILABLE | scoped to `youngdizzy/autonomous-trading-agent` only |
| MCP: Claude Code Remote (sessions, Routines) | AVAILABLE | |
| MCP: Claude Docs | AVAILABLE | not used |

## Scheduling

| Mechanism | Class | Notes |
|---|---|---|
| Routines (`create_trigger` / `send_later`) | PARTIALLY_AVAILABLE | cron-style, minimum interval normally hourly; fires a Claude session. Suitable for an hourly research/paper loop, not for sub-hour monitoring. Not configured in Foundation 1.0. |
| Session-local cron / background processes | PARTIALLY_AVAILABLE | die with the container |
| OS daemons / VPS | not used | deliberately (no server requirement) |

The autonomous loop is therefore designed as a **resumable tick** (`AutonomousLoop.tick()`): each
wake runs one full health→data→reconcile→risk→...→wait cycle from a durable checkpoint and never
assumes a long-lived process.

## Persistence

| Store | Class | Notes |
|---|---|---|
| Container filesystem | PARTIALLY_AVAILABLE | ext4, writable, but the container is **ephemeral** — reclaimed after inactivity |
| Git (commit + push) | AVAILABLE | the only durable store observed; push is subject to user authorization |
| External databases | not used | none required |

Design consequence: all durable state is append-only, hash-chained JSONL (`ati/ledger/journal.py`)
under a state directory, so it can be committed to git or copied elsewhere, and any corruption or
truncation is detected on load (fail closed).

## Secret management

| Item | Class | Notes |
|---|---|---|
| Environment-level secrets | AVAILABLE | configured by the owner in the environment settings; exposed as environment variables to new sessions |
| Brokerage credentials | intentionally absent | not requested, not configured (LIVE_TRADING=false) |
| Container env | observed | contains platform tokens (GitHub, proxy, cloud). Their **names** were listed; their values were never printed. The system must never read them. |

## Summary

| Area | Status |
|---|---|
| Local compute, Python, tests, git | AVAILABLE |
| Live market data (any provider tested) | BLOCKED — ENVIRONMENT CAPABILITY |
| Live broker connectivity | BLOCKED — ENVIRONMENT CAPABILITY (and intentionally unconfigured) |
| Programmatic Claude API | BLOCKED — CREDENTIALS (Claude-native file exchange used instead) |
| Durable persistence | PARTIALLY_AVAILABLE (git only) |
| Scheduling | PARTIALLY_AVAILABLE (hourly Routines) |

No blocked capability required weakening a security control or changing the architecture.

## Re-audit — Real Market Evidence Activation 1.0 (2026-09-28)

| Probe | Observed |
|---|---|
| DNS `api.kraken.com` | resolves (104.17.185–189.205, Cloudflare) |
| `GET https://api.kraken.com/0/public/Time` (curl, env proxy, default TLS) | `CONNECT tunnel failed, response 403` |
| `GET https://api.kraken.com/0/public/OHLC?pair=XBTUSD&interval=60` (curl) | `CONNECT tunnel failed, response 403` |
| Same OHLC request through the repository (`KrakenPublicOHLC` + `UrllibTransport`) | `ProviderUnavailable: <urlopen error Tunnel connection failed: 403 Forbidden>` — repository code reached the gateway and failed closed; nothing ingested |
| Agent proxy status log | `connect_rejected` for `api.kraken.com:443`: "gateway answered 403 to CONNECT (policy denial or upstream failure)" |

```
LIVE_CONNECTIVITY = BLOCKED — ENVIRONMENT CAPABILITY (egress policy)
```

The failure is environmental: DNS works, TLS was never reached, and the proxy rejects the tunnel
before any Kraken server is contacted. Minimum change: the environment owner adds `api.kraken.com`
to the allowed domains (session title bar → environment menu → Edit → Network access). No code
change, TLS change, or proxy workaround is required or was attempted.

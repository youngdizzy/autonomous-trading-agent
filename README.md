# autonomous-trading-agent

Autonomous trading intelligence — **Foundation 1.0**.

A research → decision → risk → execution → learning system in which Claude proposes and
deterministic software decides what is mechanically permitted. **LIVE trading is disabled.**
No brokerage credentials exist. Market data hosts are unreachable from the build environment, so
everything runnable today uses data labelled **MOCK**.

- Mission and operating principles: [`docs/MISSION.md`](docs/MISSION.md)
- Architecture and invariants: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- What is implemented / blocked / not implemented: [`docs/STATUS.md`](docs/STATUS.md)
- Security boundaries: [`docs/SECURITY.md`](docs/SECURITY.md)
- Environment audit: [`docs/ENVIRONMENT_CAPABILITY_AUDIT.md`](docs/ENVIRONMENT_CAPABILITY_AUDIT.md)

## Run

Python 3.11+, standard library only. Tests need pytest.

```bash
pip install pytest
python -m pytest                                   # full suite
python -m ati demo --state-dir /tmp/ati-demo       # MOCK research cycle + paper loop + status
python -m ati verify --state-dir /tmp/ati-demo     # verify every journal's hash chain
```

The demo's research cycle is expected to end in a **denied** promotion: the MOCK strategy passes
development walk-forward and then fails adversarial review. With no champion, the paper loop
reconciles every tick and takes no risk.

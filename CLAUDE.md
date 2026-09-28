# Working in this repository

Autonomous trading intelligence, Foundation 1.0. Read `docs/ARCHITECTURE.md` and `docs/STATUS.md`
before changing anything.

## Non-negotiables

- `LIVE_TRADING` stays `False` unless the owner explicitly asks for a reviewed change.
- Never weaken a deterministic control to make a test or demo pass. If a control blocks you, report it.
- Never label data with a stronger `DataStatus` than it has. MOCK stays MOCK.
- Never edit journals, fixtures of record, or promotion records by hand. History is append-only.
- Never give research/optimization code access to HOLDOUT data or to data overlapping a sealed period.
- Claude output only ever enters the system through `ati/agent/schema.py`.
- Use the status vocabulary in `docs/STATUS.md` exactly (IMPLEMENTED, BLOCKED, NOT IMPLEMENTED, …).

## Conventions

- Runtime is Python 3.11 standard library only. Money and prices are `Decimal`; timestamps are
  UTC-aware `datetime`. Floats only in statistics.
- Every safety failure has its own exception type in `ati/core/errors.py`; fail closed.
- New invariants need a test that would fail if the invariant were broken.

## Commands

```bash
python -m pytest
python -m ati demo --state-dir <empty dir>
python -m ati verify --state-dir <dir>
```

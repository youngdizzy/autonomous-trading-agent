# Security Boundaries

## Secrets

- Loaded only by explicit variable name (`load_secret`); the environment is never enumerated.
- Held in `SecretValue`, whose `repr`/`str` never render the value and which refuses pickling.
- Every journal append and every prompt passes through `SecretGuard.scan`, which refuses content
  containing a registered secret or a credential-shaped string (Anthropic, AWS, GitHub, Slack
  token shapes, private-key headers).
- No credentials exist in this repository; a test scans the tree for credential shapes.
- No brokerage credentials are requested or configured. `LIVE_TRADING` is a code constant.

## Prompt injection

External content (news, web pages, API text, files, messages) is data:

1. Role instructions are constants in code and always come first in a prompt.
2. External text is wrapped by `fence()`: Unicode-normalized, control characters stripped, and any
   delimiter-like sequence neutralized, so content cannot close its own fence.
3. Claude's output is parsed into a closed schema with an action whitelist
   (`PROPOSE_TRADE`, `NO_TRADE`, `REQUEST_RESEARCH`); unknown fields are rejected.
4. Even a fully hijacked output is only a request to the risk engine, which sizes it from
   deterministic limits. Heuristic injection flags are recorded but never relied upon.

## Tool and permission boundaries

- Only `ExecutionEngine` calls `Broker.submit_order` (enforced by test).
- Execution accepts only HMAC-signed, unexpired risk approvals from a per-process key.
- The system never shells out, `eval`s, or unpickles (enforced by a source scan test).
- Kill-switch release and external account adjustments require exact operator acknowledgement
  strings; no agent action maps to them.

## Filesystem boundaries

- All state lives under one state directory: journals (`*.jsonl`), `kill_switch.json`,
  `paper_venue.json`, and the reasoning exchange directory.
- File-exchange names are hashes of request ids, so ids cannot traverse paths (tested).
- Journals are append-only with hash chains; tampering, deletion, and truncation fail closed.

## Known limits

- Python cannot stop in-process code that deliberately reaches private attributes (e.g. the
  holdout vault's name-mangled slot). The boundaries protect every path the agent layer can reach
  and prevent accidental misuse; they are not a sandbox against malicious code in the process.
- The ephemeral container's filesystem is not durable; durability depends on committing or
  copying the state directory.

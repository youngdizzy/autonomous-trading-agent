"""Text-only parser for Quant Trading Vault strategy files (FMZ export format).

Observed format (brainbrick-trades/The-Quant-Trading-Vault @ c9d6fa49, every file under ``strategies/``)::

    > Name / > Author / [> Strategy Description] / [> Strategy Arguments] / > Source (<Language>) / > Detail /
    > Last Modified

Pine Script sources usually open with an FMZ ``/*backtest ... */`` block (start, end, period, basePeriod,
exchanges) — the external backtest configuration, recorded only as an EXTERNAL_CLAIM context.

Nothing here executes anything. The source block is treated as text: regular expressions, a comment stripper and a
parenthesis counter. ``json.loads`` is used only on the backtest block's ``exchanges`` value (data, not code).
Anything the parser cannot establish is recorded as ``UNKNOWN``; a file that does not follow the format raises
``VaultParseError`` (→ PARSE_FAILED), it is never guessed.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

PARSER_VERSION = "vault-md-1"

_HEADER = re.compile(r"^> (Name|Author|Strategy Description|Strategy Arguments|Source \(([A-Za-z+]+)\)|Detail|"
                     r"Last Modified)\s*$")
REQUIRED = ("Name", "Author", "Source", "Detail", "Last Modified")
_LANG = {"PineScript": "pine", "javascript": "javascript", "python": "python", "MyLanguage": "mylanguage",
         "cpp": "cpp"}

# Security scan signatures live in signatures.json (data): the system's own code never names shell or dynamic-execution
# APIs (tests/test_claude_contract_security.py), even as detection patterns.
_SIGNATURES = json.loads(Path(__file__).with_name("signatures.json").read_text(encoding="utf-8"))["patterns"]
SECURITY_PATTERNS: dict[str, tuple[str, str]] = {k: (v["category"], v["regex"]) for k, v in _SIGNATURES.items()}
UNSAFE_CATEGORIES = frozenset({"UNSAFE"})

_CLAIM_WORDS = re.compile(r"(?i)\b(return|profit|win ?rate|sharpe|drawdown|annuali[sz]ed|times|gain|accuracy)\b")
_FAMILIES = (("grid", r"\bgrid"), ("arbitrage", r"arbitrage|hedg"), ("market_making", r"market[- ]?making|spread"),
             ("periodic_investment", r"fixed investment|\bdca\b|invested every|regular"),
             ("martingale", r"martingale"), ("machine_learning", r"machine learning|neural|\bml\b|\bai\b"),
             ("breakout", r"breakout|break-out|channel|donchian"), ("reversal", r"reversal|mean[- ]revers"),
             ("trend_following", r"trend|moving average|\bema\b|\bsma\b|crossover|\bma\b"),
             ("oscillator_momentum", r"\brsi\b|macd|stoch|momentum|oscillat|\bcci\b"),
             ("volatility", r"volatility|\batr\b|bollinger"),
             ("tooling", r"library|plug-?in|template|monitor|tool|transfer|chart|example|test"))


class VaultParseError(ValueError):
    """The file does not follow the Vault format (→ PARSE_FAILED)."""


@dataclass(frozen=True)
class PineStructure:
    version: str
    header_kind: str                         # strategy | indicator | UNKNOWN
    strategy_kwargs: dict[str, str]
    inputs: dict[str, str]                   # name → default literal (text)
    assignments: dict[str, str]              # name → normalized expression (last assignment wins; all kept below)
    reassigned: tuple[str, ...]
    calls: tuple[str, ...]                   # canonical function names used outside visual statements
    uses_tr_variable: bool
    entries: tuple[dict, ...]                # {"id", "side", "condition", "text"}
    closes: tuple[dict, ...]
    exits: tuple[dict, ...]                  # strategy.exit calls: {"id", "from_entry", "kwargs", "condition", "text"}
    orders: tuple[str, ...]                  # strategy.order / close_all / cancel / risk.* texts
    unrecognized: tuple[str, ...]
    visual_statements: int


@dataclass(frozen=True)
class ParsedSpec:
    name: str
    author: str
    description: str
    arguments: tuple[tuple[str, str, str], ...]
    language: str                            # pine | javascript | python | mylanguage | cpp | UNKNOWN:<label>
    language_label: str
    source: str
    detail_url: str
    last_modified: str
    backtest: dict | None
    security: dict[str, int]
    pine: PineStructure | None
    claims: tuple[dict, ...] = field(default=())
    family: str = "UNKNOWN"


# ----------------------------------------------------------------------------------------------- markdown sections
def split_sections(text: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    order: list[str] = []
    current = None
    for line in text.splitlines():
        m = _HEADER.match(line)
        if m:
            key = "Source" if m.group(1).startswith("Source") else m.group(1)
            if key in sections:
                raise VaultParseError(f"section '{key}' appears more than once")
            sections[key] = []
            order.append(key)
            if key == "Source":
                sections["_language"] = [m.group(2)]
            current = key
        elif current is not None:
            sections[current].append(line)
    missing = [k for k in REQUIRED if k not in sections]
    if missing:
        raise VaultParseError(f"missing section(s): {', '.join(missing)}")
    return {k: "\n".join(v).strip() for k, v in sections.items()}


def _code_block(section: str) -> str:
    lines = section.splitlines()
    fences = [i for i, line in enumerate(lines) if line.strip().startswith("```")]
    if len(fences) < 2:
        raise VaultParseError("source section has no fenced code block")
    return "\n".join(lines[fences[0] + 1:fences[-1]])


def _arguments(section: str) -> tuple[tuple[str, str, str], ...]:
    rows = []
    for line in section.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 3 and cells[0] not in ("Argument", "") and not set(cells[0]) <= set("-: "):
            rows.append((cells[0], cells[1], "|".join(cells[2:])[:200]))
    return tuple(rows)


def _backtest_block(source: str) -> dict | None:
    m = re.match(r"\s*/\*backtest\s*\n(.*?)\*/", source, re.S)
    if not m:
        return None
    out: dict = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    if "exchanges" in out:
        try:
            out["exchanges"] = json.loads(out["exchanges"])
        except ValueError:
            out["exchanges"] = "UNPARSEABLE"
    return out


def security_scan(source: str) -> dict[str, int]:
    return {name: len(re.findall(rx, source)) for name, (_, rx) in SECURITY_PATTERNS.items()
            if re.search(rx, source)}


def claims(name: str, description: str, backtest: dict | None) -> tuple[dict, ...]:
    found: list[dict] = []
    if backtest:
        found.append({"status": "EXTERNAL_CLAIM", "kind": "external_backtest_configuration",
                      "detail": {k: backtest[k] for k in sorted(backtest)}})
    for text in [name] + re.split(r"(?<=[.!?])\s+|\n", description):
        t = text.strip()
        if t and re.search(r"\d", t) and _CLAIM_WORDS.search(t):
            found.append({"status": "EXTERNAL_CLAIM", "kind": "external_performance_statement", "text": t[:200]})
        if len(found) >= 12:
            break
    return tuple(found)


def family(name: str, description: str) -> str:
    text = f"{name} {description[:400]}".lower().replace("-", " ")
    for label, rx in _FAMILIES:
        if re.search(rx, text):
            return f"{label} (heuristic: name/description keywords)"
    return "UNKNOWN"


def parse(raw: bytes) -> ParsedSpec:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise VaultParseError("file is not UTF-8") from exc
    s = split_sections(text)
    label = s["_language"]
    source = _code_block(s["Source"])
    language = _LANG.get(label, f"UNKNOWN:{label}")
    backtest = _backtest_block(source) if language == "pine" else None
    name = s["Name"].strip() or "UNKNOWN"
    description = s.get("Strategy Description", "")
    return ParsedSpec(
        name=name, author=s["Author"].strip() or "UNKNOWN", description=description[:4000],
        arguments=_arguments(s.get("Strategy Arguments", "")), language=language, language_label=label,
        source=source, detail_url=s["Detail"].strip() or "UNKNOWN", last_modified=s["Last Modified"].strip() or "UNKNOWN",
        backtest=backtest, security=security_scan(source), pine=pine_structure(source) if language == "pine" else None,
        claims=claims(name, description, backtest), family=family(name, description))


# ----------------------------------------------------------------------------------------------- Pine (text only)
_VISUAL = re.compile(r"^(plot\w*|bgcolor|barcolor|fill|hline|label\.\w+|line\.\w+|box\.\w+|table\.\w+|alertcondition|"
                     r"alert|var\s+(table|label|line|box)\b)\s*\(?")
_CALL = re.compile(r"([A-Za-z_][\w.]*)\s*\(")
_INPUT = re.compile(r"^(?:var\s+)?(?:(?:int|float|bool|string|color)\s+)?(\w+)\s*=\s*input(?:\.\w+)?\s*\((.*)\)\s*$")
_ASSIGN = re.compile(r"^(?:var\s+|varip\s+)?(?:(?:int|float|bool|string|color|series\s+\w+)\s+)?(\w+)\s*(:?=)(?!=)\s*(.+)$")
_CONTROL = re.compile(r"^(if|else|for|while|switch)\b")
_PREFIXES = ("ta.", "math.", "request.")          # Pine v5 namespaces of v4 built-ins (ta.sma == sma)
_KEYWORDS = frozenset({"if", "and", "or", "not", "for", "while", "switch", "else"})
_STRING = re.compile(r"\"(?:[^\"\\]|\\.)*\"|'(?:[^'\\]|\\.)*'")


def strip_comments(source: str) -> str:
    out, i, n, in_str = [], 0, len(source), ""
    while i < n:
        c = source[i]
        if in_str:
            out.append(c)
            if c == in_str:
                in_str = ""
            i += 1
        elif c in "\"'":
            in_str = c
            out.append(c)
            i += 1
        elif source.startswith("//", i):
            while i < n and source[i] != "\n":
                i += 1
        elif source.startswith("/*", i):
            end = source.find("*/", i + 2)
            i = n if end < 0 else end + 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


def _logical_lines(code: str) -> list[tuple[int, str]]:
    lines: list[tuple[int, str]] = []
    buf, indent, depth = "", 0, 0
    for raw in code.splitlines():
        if not raw.strip():
            continue
        stripped = raw.strip()
        cont = bool(buf) and (depth > 0 or re.search(r"(\band|\bor|[-+*/?:,=<>])\s*$", buf) is not None
                              or re.match(r"^(and|or|\?|:)\b", stripped) is not None)
        if cont:
            buf += " " + stripped
        else:
            if buf:
                lines.append((indent, buf))
            buf, indent = stripped, len(raw) - len(raw.lstrip())
        depth = max(0, buf.count("(") + buf.count("[") - buf.count(")") - buf.count("]"))
    if buf:
        lines.append((indent, buf))
    return lines


def canonical_call(name: str) -> str:
    for p in _PREFIXES:
        if name.startswith(p):
            return name[len(p):]
    return name


def split_args(text: str) -> list[str]:
    parts, depth, cur, in_str = [], 0, "", ""
    for c in text:
        if in_str:
            cur += c
            in_str = "" if c == in_str else in_str
            continue
        if c in "\"'":
            in_str = c
        elif c in "([":
            depth += 1
        elif c in ")]":
            depth -= 1
        elif c == "," and depth == 0:
            parts.append(cur.strip())
            cur = ""
            continue
        cur += c
    if cur.strip():
        parts.append(cur.strip())
    return parts


def _call_args(stmt: str, fn: str) -> list[str] | None:
    i = stmt.find(fn + "(")
    if i < 0:
        return None
    depth, j = 0, i + len(fn)
    for k in range(j, len(stmt)):
        if stmt[k] == "(":
            depth += 1
        elif stmt[k] == ")":
            depth -= 1
            if depth == 0:
                return split_args(stmt[j + 1:k])
    return None


def norm_expr(expr: str) -> str:
    e = expr
    for p in _PREFIXES:
        e = e.replace(p, "")
    return re.sub(r"\s+", "", e)


def _kwargs(args: list[str]) -> tuple[list[str], dict[str, str]]:
    pos, kw = [], {}
    for a in args:
        m = re.match(r"^(\w+)\s*=(?!=)\s*(.+)$", a)
        if m:
            kw[m.group(1)] = m.group(2).strip()
        else:
            pos.append(a)
    return pos, kw


def pine_structure(source: str) -> PineStructure:
    version = (re.search(r"//@version\s*=\s*(\d+)", source) or [None, "UNKNOWN"])[1]
    lines = _logical_lines(strip_comments(source))
    header_kind, skw = "UNKNOWN", {}
    inputs, assignments, reassigned = {}, {}, []
    calls: set[str] = set()
    entries, closes, exits, orders, unrecognized = [], [], [], [], []
    visual = 0
    uses_tr = False
    stack: list[tuple[int, str]] = []            # (indent, condition) for enclosing if-blocks
    for indent, stmt in lines:
        while stack and indent <= stack[-1][0]:
            stack.pop()
        cond = " and ".join(f"({c})" for _, c in stack) if stack else ""
        head = canonical_call(stmt.split("(", 1)[0].strip()) if "(" in stmt else ""
        if head in ("strategy", "study", "indicator") and stmt.startswith(("strategy(", "study(", "indicator(")):
            header_kind = "strategy" if head == "strategy" else "indicator"
            _, skw = _kwargs(_call_args(stmt, stmt.split("(", 1)[0]) or [])
            continue
        if _VISUAL.match(stmt):
            visual += 1
            continue
        code = _STRING.sub('""', stmt)             # names inside string literals are titles, not calls
        for c in _CALL.findall(code):
            if c not in _KEYWORDS:
                calls.add(canonical_call(c))
        if re.search(r"(?<![\w.])tr(?![\w(])", code):
            uses_tr = True
        m = _INPUT.match(stmt)
        if m:
            pos, kw = _kwargs(split_args(m.group(2)))
            inputs[m.group(1)] = kw.get("defval", pos[0] if pos else "UNKNOWN")
            continue
        if stmt.startswith("strategy."):
            fn = stmt.split("(", 1)[0]
            pos, kw = _kwargs(_call_args(stmt, fn) or [])
            when = kw.get("when", "")
            full = " and ".join(x for x in (cond, f"({when})" if when else "") if x)
            item = {"fn": fn, "id": pos[0] if pos else kw.get("id", "UNKNOWN"), "condition": full,
                    "kwargs": kw, "positional": pos[1:], "text": stmt[:200]}
            if fn == "strategy.entry":
                side = (pos[1] if len(pos) > 1 else kw.get("direction", kw.get("long", "UNKNOWN"))).replace(" ", "")
                item["side"] = {"strategy.long": "long", "true": "long", "strategy.short": "short",
                                "false": "short"}.get(side, "UNKNOWN")
                entries.append(item)
            elif fn == "strategy.close":
                closes.append(item)
            elif fn == "strategy.exit":
                item["from_entry"] = kw.get("from_entry", "")
                exits.append(item)
            else:
                orders.append(stmt[:200])
            continue
        c = _CONTROL.match(stmt)
        if c:
            if c.group(1) == "if":
                stack.append((indent, stmt[2:].strip()))
            elif c.group(1) == "else":
                stack.append((indent, "else"))       # an else-branch condition is not reconstructed; marked
            else:
                unrecognized.append(stmt[:200])
                stack.append((indent, f"loop:{stmt[:60]}"))
            continue
        a = _ASSIGN.match(stmt)
        if a:
            name, op, expr = a.groups()
            if name in assignments or op == ":=":
                reassigned.append(name)
            assignments[name] = norm_expr(expr)
            continue
        unrecognized.append(stmt[:200])
    return PineStructure(version, header_kind, skw, inputs, assignments, tuple(sorted(set(reassigned))),
                         tuple(sorted(calls)), uses_tr, tuple(entries), tuple(closes), tuple(exits), tuple(orders),
                         tuple(unrecognized), visual)

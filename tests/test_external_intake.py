"""Phase 5 — external strategy intake (ati.intake): provenance, safe parsing, normalization, compatibility,
claim separation, research-universe accounting, Factory integration, holdout and trading isolation.

Every strategy file here is a hand-written TEST FIXTURE in the Vault's format, committed to a throw-away git
repository; no Vault source is copied into this repository. Datasets are MOCK and labelled as such.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
from datetime import timedelta
from pathlib import Path

import pytest

from ati.company import factory
from ati.core.errors import ExternalSourceError
from ati.intake import normalize
from ati.intake.corpus import ExternalIntake, research_external
from ati.intake.source import GitCorpus, SourceArtifact
from ati.ledger.journal import decode
from ati.market.models import Timeframe
from ati.research.adversarial import AdversarialPolicy
from ati.research.protocols import REGISTRY
from ati.strategies.base import LOGIC_REGISTRY
from ati.validation.promotion import PromotionPolicy
from tests.helpers import mock_dataset
from tests.rig import make_system

REPO = "fixture/vault"
MECHANICS = dict(adversarial_policy=AdversarialPolicy(allow_non_market_data=True),
                 promotion_policy=PromotionPolicy(allow_mock_evidence=True))
INTAKE_DIR = Path(__file__).resolve().parents[1] / "ati" / "intake"

HEADER_1H_SPOT = """/*backtest
start: 2023-01-01 00:00:00
end: 2023-06-01 00:00:00
period: 1h
basePeriod: 15m
exchanges: [{"eid":"Binance","currency":"BTC_USDT"}]
*/
"""
TEMPLATE = HEADER_1H_SPOT + """//@version=4
strategy("Fixture SMA crossover", overlay=true)
fastLen = input(10, minval=1)
slowLen = input(50, minval=2)
F = sma(close, fastLen)
S = sma(close, slowLen)
A = sma(tr(true), 14)
T = 0.0
T := strategy.position_size > 0 ? max(nz(T[1]), close - 3.0 * A) : close - 3.0 * A
strategy.entry("L", strategy.long, when = F > S)
strategy.close("L", when = F <= S)   // exit when fast <= slow
strategy.exit("LS", from_entry = "L", stop = T)
plot(F, title="Fast (Profit Line)")
"""


def spec(source: str, language: str = "PineScript", name: str = "Fixture", description: str = "Fixture.") -> str:
    fence = {"PineScript": "pinescript", "javascript": "javascript", "python": "python"}.get(language, "")
    return (f"\n> Name\n\n{name}\n\n> Author\n\nfixture-author\n\n> Strategy Description\n\n{description}\n\n"
            f"> Strategy Arguments\n\n\n\n|Argument|Default|Description|\n|----|----|----|\n|v_input_1|10|fast|\n\n\n"
            f"> Source ({language})\n\n``` {fence}\n{source}\n```\n\n> Detail\n\nhttps://example.invalid/strategy/1\n\n"
            f"> Last Modified\n\n2024-01-01 00:00:00\n")


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


def commit(repo: Path, files: dict[str, str | bytes], message: str = "c") -> str:
    if not (repo / ".git").exists():
        repo.mkdir(parents=True, exist_ok=True)
        git(repo, "init", "-q")
        git(repo, "config", "user.email", "fixture@example.invalid")
        git(repo, "config", "user.name", "fixture")
    for name, content in files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content if isinstance(content, bytes) else content.encode("utf-8"))
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", message)
    return git(repo, "rev-parse", "HEAD")


def corpus(repo: Path, sha: str) -> GitCorpus:
    """Operator steps, done here by the test: check out the commit and write its `git ls-tree -r -z` listing."""
    git(repo, "checkout", "-q", sha)
    manifest = repo.parent / f"{repo.name}-{sha[:12]}.manifest"
    manifest.write_bytes(subprocess.run(["git", "-C", str(repo), "ls-tree", "-r", "-z", sha], check=True,
                                        capture_output=True).stdout)
    return GitCorpus(repo, REPO, sha, manifest)


def artifact(tmp_path: Path, source: str, **kw) -> SourceArtifact:
    repo = tmp_path / "corpus"
    sha = commit(repo, {"strategies/x.md": spec(source, **kw)})
    return corpus(repo, sha).read("strategies/x.md")


def record(tmp_path, source, **kw) -> dict:
    return normalize.build_record(artifact(tmp_path, source, **kw))


def states(rec) -> set[str]:
    return {r["state"] for r in rec["reasons"]}


# ------------------------------------------------------------------------------------------------ provenance A–E
class TestProvenance:
    def test_A_same_source_same_hash_and_identity(self, tmp_path):
        repo = tmp_path / "c"
        sha = commit(repo, {"strategies/a.md": spec(TEMPLATE)})
        one, two = corpus(repo, sha).read("strategies/a.md"), corpus(repo, sha).read("strategies/a.md")
        assert one.source_hash == two.source_hash and one.external_strategy_id == two.external_strategy_id
        assert one.git_blob == git(repo, "rev-parse", f"{sha}:strategies/a.md")      # the id GitHub serves

    def test_B_changed_source_changes_hash(self, tmp_path):
        repo = tmp_path / "c"
        a = commit(repo, {"strategies/a.md": spec(TEMPLATE)})
        b = commit(repo, {"strategies/a.md": spec(TEMPLATE.replace("50", "60"))})
        old, new = corpus(repo, a).read("strategies/a.md"), corpus(repo, b).read("strategies/a.md")
        assert old.source_hash != new.source_hash and old.external_strategy_id != new.external_strategy_id

    def test_C_identity_is_not_a_filename_or_name(self, tmp_path):
        repo = tmp_path / "c"
        sha = commit(repo, {"strategies/a.md": spec(TEMPLATE, name="Same"), "strategies/b.md": spec(TEMPLATE, name="Same")
                            .replace("Fixture.", "Other."), "strategies/c.md": spec(TEMPLATE, name="Same")})
        a, b, c = corpus(repo, sha).read_many(["strategies/a.md", "strategies/b.md", "strategies/c.md"])
        assert a.source_hash != b.source_hash and a.external_strategy_id != b.external_strategy_id  # content differs
        assert a.source_hash == c.source_hash and a.external_strategy_id != c.external_strategy_id  # path differs

    def test_D_commit_is_preserved_and_modified_checkouts_are_refused(self, tmp_path):
        repo = tmp_path / "c"
        a = commit(repo, {"strategies/a.md": spec(TEMPLATE)})
        view = corpus(repo, a)
        rec = normalize.build_record(view.read("strategies/a.md"))
        assert rec["source_commit"] == a and rec["compatibility_status"] == "COMPATIBLE"
        assert rec["source_url"].endswith(f"/blob/{a}/strategies/a.md")
        (repo / "strategies/a.md").write_text("tampered in the working tree")
        with pytest.raises(ExternalSourceError):
            view.read("strategies/a.md")                          # bytes are no longer commit a's blob
        b = commit(repo, {"strategies/a.md": spec(TEMPLATE.replace("50", "60"))})
        with pytest.raises(ExternalSourceError):
            view.read("strategies/a.md")                          # checkout at b, manifest of a: refused
        assert corpus(repo, b).read("strategies/a.md").commit == b
        with pytest.raises(ExternalSourceError):
            view.read("strategies/missing.md")                    # not in the commit's tree
        with pytest.raises(ExternalSourceError):
            GitCorpus(repo, REPO, "HEAD", tmp_path / f"c-{a[:12]}.manifest")   # symbolic refs are not identities

    def test_E_AD_intake_state_survives_restart(self, tmp_path):
        s, _ = make_system(tmp_path / "st")
        rec = ExternalIntake(s.research_journal).ingest(artifact(tmp_path, TEMPLATE))
        s2, _ = make_system(tmp_path / "st")
        again = ExternalIntake(s2.research_journal)
        assert again.records[rec["external_strategy_id"]] == rec
        assert again.sources[rec["external_strategy_id"]]["source_commit"] == rec["source_commit"]

    def test_unsafe_paths_are_refused(self, tmp_path):
        repo = tmp_path / "c"
        sha = commit(repo, {"strategies/a.md": spec(TEMPLATE)})
        for bad in ("../etc/passwd", "/abs.md", "a\nb.md", ""):
            with pytest.raises(ExternalSourceError):
                corpus(repo, sha).read(bad)


# ------------------------------------------------------------------------------------------------ parsing F–I
class TestParsing:
    def test_F_supported_strategy_parses(self, tmp_path):
        rec = record(tmp_path, TEMPLATE)
        assert rec["compatibility_status"] == "COMPATIBLE", rec["reasons"]
        assert rec["source_language"] == "pine" and rec["timeframe_scope"] == "1h"
        assert rec["mapping"] == {"kind": "ma_crossover", "timeframe": "1h",
                                  "params": {"fast": 10, "slow": 50, "atr_period": 14, "stop_atr": 3.0}}

    @pytest.mark.parametrize("text", [
        "no sections at all",
        spec(TEMPLATE).replace("> Author", "> Writer"),                          # missing section
        spec(TEMPLATE).replace("> Detail", "> Name\n\nagain\n\n> Detail"),       # duplicated section
        spec(TEMPLATE).replace("```", "~~~"),                                    # no fenced source
    ])
    def test_G_malformed_is_PARSE_FAILED(self, tmp_path, text):
        repo = tmp_path / "c"
        sha = commit(repo, {"strategies/x.md": text})
        rec = normalize.build_record(corpus(repo, sha).read("strategies/x.md"))
        assert rec["compatibility_status"] == rec["normalization_status"] == "PARSE_FAILED"
        assert rec["tradetown_strategy_id"] is None

    def test_G_non_utf8_is_PARSE_FAILED(self, tmp_path):
        repo = tmp_path / "c"
        sha = commit(repo, {"strategies/x.md": b"\xff\xfe" + spec(TEMPLATE).encode()})
        assert normalize.build_record(corpus(repo, sha).read("strategies/x.md"))["compatibility_status"] \
            == "PARSE_FAILED"

    def test_H_parser_never_executes_strategy_code(self, tmp_path):
        mark = tmp_path / "EXECUTED"
        py = f"import os\nopen({str(mark)!r}, 'w').write('x')\n__import__('os').system('touch {mark}')\n"
        js = f"require('child_process').execSync('touch {mark}'); eval('1')"
        for src, lang in ((py, "python"), (js, "javascript"), (TEMPLATE + f"\n// {py}", "PineScript")):
            rec = record(tmp_path / lang, src, language=lang)
            assert not mark.exists()
            if lang != "PineScript":
                assert rec["compatibility_status"] == "UNSAFE", rec["reasons"]

    def test_H_intake_code_has_no_dynamic_execution(self):
        for path in INTAKE_DIR.glob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    assert node.func.id not in ("eval", "exec", "compile", "__import__"), path
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    names = [a.name for a in node.names] + [getattr(node, "module", None) or ""]
                    assert not any(n.split(".")[0] in ("importlib", "runpy", "pickle", "marshal", "ctypes")
                                   for n in names), path
                    assert not any(n.split(".")[0] in ("subprocess", "os", "shutil") for n in names), path
        # detection signatures are data (signatures.json), so the code itself never names shell/dynamic-exec APIs
        sig = json.loads((INTAKE_DIR / "signatures.json").read_text())["patterns"]
        assert {"dynamic_code", "shell", "credentials"} <= {k for k, v in sig.items() if v["category"] == "UNSAFE"}

    def test_I_unsupported_language_is_explicit(self, tmp_path):
        rec = record(tmp_path, "function main() { Log(1) }", language="javascript")
        assert rec["compatibility_status"] == "UNSUPPORTED_LANGUAGE"
        assert rec["source_language"] == "javascript" and rec["tradetown_strategy_id"] is None
        rec = record(tmp_path / "b", "function main() { exchange.Buy(-1, 1) }", language="javascript")
        assert "order_calls" in rec["security_findings"] and "execution code present" in rec["reasons"][0]["reason"]


# ------------------------------------------------------------------------------------------------ normalization J–M
class TestNormalization:
    def test_J_M_mapping_is_deterministic(self, tmp_path):
        art = artifact(tmp_path / "a", TEMPLATE)
        a, again = normalize.build_record(art), normalize.build_record(art)
        assert a == again and a["record_hash"] == again["record_hash"]
        assert normalize.definition(a).definition_hash == a["normalized_strategy_hash"]
        # the same text in another repository/commit is another source identity with the same mapped rule
        b = record(tmp_path / "b", TEMPLATE)
        assert b["mapping"] == a["mapping"] and b["source_hash"] == a["source_hash"]
        assert normalize.definition(b).behavior_fingerprint != "" and \
            normalize.definition(b).params == normalize.definition(a).params

    @pytest.mark.parametrize("change, expected", [
        (("F > S)", "F > S and close > open)"), "PARTIALLY_COMPATIBLE"),               # extra entry condition
        (("when = F > S", "when = crossover(F, S)"), "PARTIALLY_COMPATIBLE"),          # event, not state
        (("F <= S", "F < S"), "PARTIALLY_COMPATIBLE"),                                 # equality handled differently
        (("sma(tr(true), 14)", "atr(14)"), "UNSUPPORTED_INDICATOR"),                   # Wilder RMA ≠ simple mean
        (("max(nz(T[1]), close - 3.0 * A)", "close - 3.0 * A"), "PARTIALLY_COMPATIBLE"),  # no ratchet
        (("plot(F", "g(x) => x * 2\nplot(g(F)"), "SEMANTICS_UNCERTAIN"),               # user function
    ])
    def test_K_changed_meaning_is_never_COMPATIBLE(self, tmp_path, change, expected):
        assert change[0] in TEMPLATE
        rec = record(tmp_path, TEMPLATE.replace(*change))
        assert rec["compatibility_status"] == expected, rec["reasons"]
        assert rec["tradetown_strategy_id"] is None and rec["normalized_strategy_hash"] is None
        with pytest.raises(ValueError):
            normalize.definition(rec)

    def test_K_description_does_not_override_the_source(self, tmp_path):
        rec = record(tmp_path, TEMPLATE, description="Goes SHORT when fast crosses below slow. 900% return.")
        assert rec["compatibility_status"] == "COMPATIBLE"            # the source block is the specification of record

    def test_L_normalization_preserves_source_identity(self, tmp_path):
        art = artifact(tmp_path, TEMPLATE)
        rec = normalize.build_record(art)
        assert {k: rec[k] for k in art.identity()} == art.identity()
        d = normalize.definition(rec)
        assert d.strategy_id == rec["tradetown_strategy_id"] and art.external_strategy_id in d.description
        assert d.kind in LOGIC_REGISTRY and d.timeframe is Timeframe.H1

    def test_M_tampered_mapping_is_refused(self, tmp_path):
        rec = record(tmp_path, TEMPLATE)
        forged = rec | {"mapping": rec["mapping"] | {"params": rec["mapping"]["params"] | {"fast": 5}}}
        with pytest.raises(ValueError):
            normalize.definition(forged)                               # normalized hash no longer matches


# ------------------------------------------------------------------------------------------------ compatibility N–R
class TestCompatibility:
    @pytest.mark.parametrize("change, state", [
        (("period: 1h", "period: 1d"), "UNSUPPORTED_TIMEFRAME"),
        (("period: 1h\n", ""), "UNSUPPORTED_TIMEFRAME"),
        (('"eid":"Binance"', '"eid":"Futures_Binance"'), "UNSUPPORTED_MARKET"),
        (("BTC_USDT", "SOL_USDT"), "UNSUPPORTED_MARKET"),
        (("strategy.long", "strategy.short"), "UNSUPPORTED_EXECUTION_MODEL"),
        (("overlay=true", "overlay=true, pyramiding=3"), "UNSUPPORTED_EXECUTION_MODEL"),
        (("overlay=true", "overlay=true, process_orders_on_close=true"), "UNSUPPORTED_EXECUTION_MODEL"),
        (("S = sma(close, slowLen)", "S = sma(close, slowLen)\nR = rsi(close, 14)"), "UNSUPPORTED_INDICATOR"),
        (('strategy("Fixture SMA crossover", overlay=true)', 'study("Fixture")'), "INCOMPATIBLE"),
        (("input(10", "input(80"), "INCOMPATIBLE"),                                     # fast >= slow: outside bounds
    ])
    def test_N_O_P_Q_unsupported_constructs_are_rejected(self, tmp_path, change, state):
        assert change[0] in TEMPLATE
        rec = record(tmp_path, TEMPLATE.replace(*change))
        assert state in states(rec), rec["reasons"]
        assert rec["compatibility_status"] != "COMPATIBLE" and rec["tradetown_strategy_id"] is None

    def test_R_compatible_candidate_enters_the_existing_registry_through_the_workflow(self, tmp_path):
        s, _ = make_system(tmp_path / "st")
        intake = ExternalIntake(s.research_journal)
        rec = intake.ingest(artifact(tmp_path, TEMPLATE))
        assert not any(k.startswith("ext-") for k in s.strategies.definitions())   # intake never registers directly
        result = research_external(s, intake, rec["external_strategy_id"], mock_dataset(3200),
                                   protocol=REGISTRY["REAL-PROTOCOL-001"], **MECHANICS)
        assert result.status == "COMPLETED", result.reasons
        prereg = decode(next(s.research_journal.entries("preregistration")).payload)["prereg"]
        assert prereg["strategy_hash"] == rec["normalized_strategy_hash"]
        assert prereg["strategy_key"] == f"{rec['tradetown_strategy_id']}@v1/1h"
        registered = [k for k in s.strategies.definitions() if k.startswith("ext-")]
        if result.dev_verdict.value == "PASS":
            assert result.challenger_key in registered
        else:
            assert registered == [] and result.memory_entry_id is not None     # rejected, remembered, not registered


# ------------------------------------------------------------------------------------------------ claims S–T
CLAIMY = "This strategy returned 900% with a 99% win rate and a Sharpe of 7. Profitable every year."


class TestClaims:
    def test_S_claims_are_recorded_only_as_EXTERNAL_CLAIM(self, tmp_path):
        rec = record(tmp_path, TEMPLATE, description=CLAIMY)
        assert rec["external_claims"] and all(c["status"] == "EXTERNAL_CLAIM" for c in rec["external_claims"])
        assert any(c["kind"] == "external_backtest_configuration" for c in rec["external_claims"])
        assert any("900%" in c.get("text", "") for c in rec["external_claims"])

    def test_S_claims_never_become_evidence_or_change_readiness(self, tmp_path):
        from ati.market.accumulate import SERIES
        from ati.market.health import readiness_table

        s, _ = make_system(tmp_path / "st")
        before = json.dumps(readiness_table(s, SERIES), default=str, sort_keys=True)
        evidence_before = sum(1 for _ in s.evidence.journal.entries())
        ExternalIntake(s.research_journal).ingest(artifact(tmp_path, TEMPLATE, description=CLAIMY))
        assert json.dumps(readiness_table(s, SERIES), default=str, sort_keys=True) == before
        assert sum(1 for _ in s.evidence.journal.entries()) == evidence_before      # nothing registered as evidence
        types = {e.type for e in s.research_journal.entries()}
        assert types <= {"external_source", "external_intake", "strategy_lifecycle"}, types

    def test_T_no_evaluation_code_reads_intake_records(self):
        root = INTAKE_DIR.parent
        for path in root.rglob("*.py"):
            if path.parent == INTAKE_DIR or path.name == "cli.py":
                continue
            text = path.read_text()
            assert not re.search(r"(from|import)\s+ati\.intake", text), path
            for kind in ("external_intake", "external_source", "external_claims"):
                assert kind not in text, (path, kind)

    def test_T_claims_do_not_change_the_research_contract(self, tmp_path):
        from ati.data.dataset import _clear_sealed_ranges_for_tests

        runs = []
        for label, desc in (("plain", "Plain."), ("claimy", CLAIMY)):
            _clear_sealed_ranges_for_tests()        # two independent companies; sealed periods are process-wide
            s, _ = make_system(tmp_path / label)
            intake = ExternalIntake(s.research_journal)
            rec = intake.ingest(artifact(tmp_path / label, TEMPLATE, description=desc))
            r = research_external(s, intake, rec["external_strategy_id"], mock_dataset(3200),
                                  protocol=REGISTRY["REAL-PROTOCOL-001"], **MECHANICS)
            prereg = decode(next(s.research_journal.entries("preregistration")).payload)["prereg"]
            runs.append((r.dev_verdict, prereg["criteria"], prereg["min_trades"],
                         [decode(e.payload)["metrics"] for e in s.research_journal.entries("experiment")]))
        assert runs[0] == runs[1]                  # identical verdict, criteria and metrics whatever the claims say
        assert [tuple(c.values()) for c in runs[0][1]] == [tuple(c) for c in REGISTRY["REAL-PROTOCOL-001"].criteria]


# ------------------------------------------------------------------------------------------------ research U–X
class TestResearchUniverse:
    def _pilot(self, tmp_path, s, n=6):
        repo = tmp_path / "c"
        files = {f"strategies/p{i}.md": spec(TEMPLATE.replace("input(10", f"input({10 + i}")) for i in range(n)}
        files |= {"strategies/j0.md": spec("function main(){}", language="javascript"),
                  "strategies/README.md": "index"}
        sha = commit(repo, files)
        corpus_ = corpus(repo, sha)
        intake = ExternalIntake(s.research_journal)
        chosen, census = intake.select_pilot(corpus_, {"PineScript": 3, "javascript": 1}, "salt")
        for a in corpus_.read_many(chosen):
            intake.ingest(a)
        return intake, corpus_, chosen, census, intake.record_universe(corpus_, "b1", chosen, census, "salt",
                                                                        {"PineScript": 3, "javascript": 1})

    def test_U_V_imported_candidates_use_the_existing_workflow_with_lineage(self, tmp_path):
        s, _ = make_system(tmp_path / "st")
        intake, view, chosen, census, u = self._pilot(tmp_path, s)
        xid = next(c["external_strategy_id"] for c in u["candidates"] if c["compatibility_status"] == "COMPATIBLE")
        research_external(s, intake, xid, mock_dataset(3200), protocol=REGISTRY["REAL-PROTOCOL-001"], **MECHANICS)
        prereg = decode(next(s.research_journal.entries("preregistration")).payload)["prereg"]
        assert prereg["observation_refs"] == [xid, intake.records[xid]["record_hash"]]            # V: lineage
        assert xid in prereg["statement"] or intake.records[xid]["source_path"] in prereg["statement"]
        new_types = {e.type for e in s.research_journal.entries()} - {"external_source", "external_intake",
                                                                        "research_universe"}
        existing = set()                               # record types written by pre-existing (non-intake) modules
        for path in INTAKE_DIR.parent.rglob("*.py"):
            if path.parent != INTAKE_DIR:
                existing |= set(re.findall(r'\.append\(\s*"(\w+)"', path.read_text()))
        assert new_types and new_types <= existing, new_types - existing                        # U: no parallel engine
        assert intake.universe_summary(u["universe_id"])["researched"] == 1

    def test_W_universe_counts_are_recorded(self, tmp_path):
        s, _ = make_system(tmp_path / "st")
        intake, view, chosen, census, u = self._pilot(tmp_path, s)
        assert census == {"corpus_files": 7, "by_language": {"PineScript": 6, "javascript": 1}}
        assert u["number_of_candidates_considered"] == 4 and len(chosen) == 4
        assert u["number_of_candidates_accepted"] == 3 and u["number_of_candidates_rejected"] == 1
        assert u["corpus"]["commit"] == view.commit and "blind to" in u["selection_method"]
        s2, _ = make_system(tmp_path / "st")                                                    # survives restart
        assert ExternalIntake(s2.research_journal).universes[u["universe_id"]] == u

    def test_X_attempts_carry_the_size_of_the_search(self, tmp_path):
        s, _ = make_system(tmp_path / "st")
        intake, view, chosen, census, u = self._pilot(tmp_path, s)
        xids = [c["external_strategy_id"] for c in u["candidates"] if c["compatibility_status"] == "COMPATIBLE"]
        research_external(s, intake, xids[0], mock_dataset(3200), protocol=REGISTRY["REAL-PROTOCOL-001"], **MECHANICS)
        eth = mock_dataset(3200, symbol="ETH/USD")      # BTC/USD's holdout is sealed now; a second series is needed
        second_run = research_external(s, intake, xids[1], eth, protocol=REGISTRY["REAL-PROTOCOL-003"], **MECHANICS)
        assert second_run.status == "COMPLETED", second_run.reasons
        again = research_external(s, intake, xids[0], mock_dataset(3200), protocol=REGISTRY["REAL-PROTOCOL-001"],
                                  **MECHANICS)
        assert again.status == "NOT_RUN"               # never re-run on the sealed data (nor under the same id)
        second = f"H-{xids[1]}-REAL-PROTOCOL-003"
        a = factory.attempts(s, second)
        assert a["external_universes_before"] == 1 and a["external_candidates_considered_before"] == 4
        assert a["root_hypotheses_tested_before"] == 1                    # the first external candidate counted
        assert intake.universe_summary(u["universe_id"])["researched"] == 2

    def test_selection_is_blind_to_content_and_claims(self, tmp_path):
        s, _ = make_system(tmp_path / "st")
        _, _, chosen, _, _ = self._pilot(tmp_path / "one", s)
        s2, _ = make_system(tmp_path / "st2")
        repo = tmp_path / "two" / "c"
        files = {f"strategies/p{i}.md": spec(TEMPLATE, description=CLAIMY * (i + 1)) for i in range(6)}
        files |= {"strategies/j0.md": spec("x", language="javascript"), "strategies/README.md": "index"}
        again, _ = ExternalIntake(s2.research_journal).select_pilot(corpus(repo, commit(repo, files)),
                                                                   {"PineScript": 3, "javascript": 1}, "salt")
        assert again == chosen                                           # same paths + salt → same pilot

    def test_pilot_cli_refuses_more_than_25(self, tmp_path, capsys):
        from ati.cli import main

        assert main(["intake-vault", "--state-dir", str(tmp_path / "st"), "--data", "mock", "--corpus", str(tmp_path),
                     "--manifest", str(tmp_path / "m"), "--commit", "0" * 40, "--per-language", "PineScript=26"]) == 2
        assert "at most 25" in capsys.readouterr().out


# ------------------------------------------------------------------------------------------------ holdout Y–Z
class TestHoldout:
    def test_Y_boundary_is_the_protocols_and_development_excludes_the_holdout(self, tmp_path):
        s, _ = make_system(tmp_path / "st")
        intake = ExternalIntake(s.research_journal)
        rec = intake.ingest(artifact(tmp_path, TEMPLATE))
        full = mock_dataset(3200)
        research_external(s, intake, rec["external_strategy_id"], full, protocol=REGISTRY["REAL-PROTOCOL-001"],
                          **MECHANICS)
        seal = decode(next(s.research_journal.entries("holdout_sealed")).payload)
        proto = REGISTRY["REAL-PROTOCOL-001"]
        assert seal["boundary"] == full.candles[int(len(full) * (1 - proto.holdout_fraction))].open_time
        prereg = decode(next(s.research_journal.entries("preregistration")).payload)["prereg"]
        assert prereg["dev_dataset_id"] != full.dataset_id                 # development partition, not the full set
        # the sealed period is now off-limits: the same data cannot be researched again
        other = normalize.build_record(artifact(tmp_path / "o", TEMPLATE.replace("input(10", "input(12")))
        intake.records[other["external_strategy_id"]] = other
        r = research_external(s, intake, other["external_strategy_id"], full, protocol=proto, **MECHANICS)
        assert r.status == "NOT_RUN" and any("holdout" in x.lower() for x in r.reasons), r.reasons

    def test_Z_intake_never_reads_holdout_or_results(self):
        for path in INTAKE_DIR.glob("*.py"):
            code = re.sub(r'""".*?"""', "", path.read_text(), flags=re.S)   # docstrings describe; code must not touch
            for word in ("holdout", "HoldoutVault", "\"experiment\"", "promotion_decision", "adversarial_report"):
                if path.name == "corpus.py" and word == "holdout":
                    continue      # corpus.py only *passes* data to the workflow; see assertion below
                assert word not in code, (path.name, word)
        corpus_code = (INTAKE_DIR / "corpus.py").read_text()
        assert "entries(\"preregistration\")" in corpus_code and "entries(\"experiment\")" not in corpus_code

    def test_Y_mismatched_dimension_is_refused(self, tmp_path):
        s, _ = make_system(tmp_path / "st")
        intake = ExternalIntake(s.research_journal)
        rec = intake.ingest(artifact(tmp_path, TEMPLATE))
        with pytest.raises(ValueError):
            research_external(s, intake, rec["external_strategy_id"], mock_dataset(3200),
                              protocol=REGISTRY["REAL-PROTOCOL-002"], **MECHANICS)       # 1h rule, 4h protocol


# ------------------------------------------------------------------------------------------------ isolation AA–AC
class TestTradingIsolation:
    FORBIDDEN = ("ati.execution", "ati.risk", "ati.agent", "ati.company.control", "ati.company.autonomy",
                 "ati.system", "ati.cli")

    def test_AA_AB_AC_intake_imports_no_trading_module(self):
        for path in INTAKE_DIR.glob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom):
                    mods = [node.module or ""]
                elif isinstance(node, ast.Import):
                    mods = [a.name for a in node.names]
                else:
                    continue
                for m in mods:
                    assert not m.startswith(self.FORBIDDEN), (path.name, m)
            text = path.read_text()
            for word in ("Broker", "place_order", "submit", "RiskEngine", "KillSwitch", "Gatekeeper", "LIVE_TRADING"):
                assert word not in text, (path.name, word)

    def test_AC_only_registered_logic_can_be_produced(self, tmp_path):
        rec = record(tmp_path, TEMPLATE)
        forged = rec | {"mapping": rec["mapping"] | {"kind": "external_code"}}
        with pytest.raises((ValueError, KeyError)):
            normalize.definition(forged)

    def test_tampered_journal_record_fails_closed(self, tmp_path):
        s, _ = make_system(tmp_path / "st")
        ExternalIntake(s.research_journal).ingest(artifact(tmp_path, TEMPLATE))
        entry = next(s.research_journal.entries("external_intake"))
        payload = decode(entry.payload)
        payload["record"]["compatibility_status"] = "COMPATIBLE"
        payload["record"]["external_claims"] = [{"status": "EVIDENCE"}]
        s.research_journal.append("external_intake", payload)                  # a forged record, appended
        with pytest.raises(ExternalSourceError):
            ExternalIntake(s.research_journal)


# ------------------------------------------------------------------------------------------------ idempotence AE–AF
class TestIdempotence:
    def test_AE_duplicate_import_is_idempotent(self, tmp_path):
        s, _ = make_system(tmp_path / "st")
        art = artifact(tmp_path, TEMPLATE)
        intake = ExternalIntake(s.research_journal)
        first = intake.ingest(art)
        n = sum(1 for _ in s.research_journal.entries())
        assert intake.ingest(art) == first and ExternalIntake(s.research_journal).ingest(art) == first
        assert sum(1 for _ in s.research_journal.entries()) == n

    def test_AF_changed_source_is_a_new_identity_and_the_old_one_stays(self, tmp_path):
        s, _ = make_system(tmp_path / "st")
        repo = tmp_path / "c"
        a = commit(repo, {"strategies/a.md": spec(TEMPLATE)})
        b = commit(repo, {"strategies/a.md": spec(TEMPLATE.replace("input(10", "input(20"))})
        intake = ExternalIntake(s.research_journal)
        old = intake.ingest(corpus(repo, a).read("strategies/a.md"))
        new = intake.ingest(corpus(repo, b).read("strategies/a.md"))
        assert old["external_strategy_id"] != new["external_strategy_id"]
        restarted = ExternalIntake(s.research_journal)
        assert restarted.records[old["external_strategy_id"]] == old and restarted.records[new["external_strategy_id"]] == new
        assert restarted.verify(corpus(repo, a)) == [] and restarted.verify(corpus(repo, b)) == []

    def test_AF_same_location_different_bytes_is_refused(self, tmp_path):
        s, _ = make_system(tmp_path / "st")
        art = artifact(tmp_path, TEMPLATE)
        intake = ExternalIntake(s.research_journal)
        intake.ingest(art)
        raw = art.raw.replace(b"input(10", b"input(11")
        import hashlib
        forged = SourceArtifact(art.repository, art.commit, art.path, art.url, art.git_blob,
                                hashlib.sha256(raw).hexdigest(), len(raw), raw)
        with pytest.raises(ExternalSourceError):
            intake.ingest(forged)
        with pytest.raises(ExternalSourceError):                          # bytes that do not match their own hash
            intake.ingest(SourceArtifact(art.repository, art.commit, "strategies/y.md", art.url, art.git_blob,
                                         art.source_hash, art.size, raw))


def test_pilot_bound_and_holdout_timedelta_sanity():
    # the research windows used above are the protocol's own, never reduced for external candidates
    p = REGISTRY["REAL-PROTOCOL-001"]
    assert p.min_candles == 3000 and p.train_bars and p.test_bars and timedelta(hours=1) == p.timeframe.delta

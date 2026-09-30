"""CLI tests. None of these touch the network (``ingest-kraken`` is exercised manually; it is BLOCKED here)."""

from ati.cli import main


def test_demo_then_verify(tmp_path, capsys):
    state = tmp_path / "demo"
    assert main(["demo", "--state-dir", str(state), "--ticks", "2"]) == 0
    out = capsys.readouterr().out
    assert "RESEARCH (MOCK data)" in out and "LIVE_TRADING=False" in out
    assert main(["verify", "--state-dir", str(state)]) == 0
    assert "CORRUPT" not in capsys.readouterr().out
    assert main(["demo", "--state-dir", str(state)]) == 2  # refuses to reuse state


def test_verify_detects_tampering(tmp_path, capsys):
    state = tmp_path / "demo"
    main(["demo", "--state-dir", str(state), "--ticks", "1"])
    path = state / "research.jsonl"
    path.write_text(path.read_text().replace('"FAIL"', '"PASS"', 1))
    assert main(["verify", "--state-dir", str(state)]) == 1
    assert "CORRUPT   research.jsonl" in capsys.readouterr().out


def test_research_real_without_real_data_is_not_run(tmp_path, capsys):
    state = tmp_path / "real"
    assert main(["research-real", "--state-dir", str(state)]) == 4
    out = capsys.readouterr().out
    assert "REAL RESEARCH RUN = NOT_RUN" in out and "no archived REAL data" in out

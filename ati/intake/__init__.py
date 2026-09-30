"""External strategy intake — the trust boundary between an external strategy corpus and TradeTown research.

    EXTERNAL CORPUS (checkout + operator-supplied `git ls-tree -r -z` of an exact commit)
        → UNTRUSTED SOURCE ARTIFACT      ati.intake.source   bytes + git blob id + SHA-256; never executed
        → STRATEGY INTAKE                ati.intake.vault    text-only parsing, security scan, claim extraction
        → NORMALIZED EXTERNAL RECORD     ati.intake.normalize deterministic compatibility decision
        → TRADETOWN CANDIDATE            ati.intake.corpus   journaled; a COMPATIBLE record yields a StrategyDefinition
        → EXISTING RESEARCH WORKFLOW     ati.research.workflow.run_research_cycle (pre-registration, walk-forward,
                                         adversarial, sealed holdout, promotion gate — unchanged)

Structural guarantees (enforced by tests that inspect this package's imports and calls):
  * nothing here imports execution, broker, risk, agent or company-control modules;
  * nothing here evaluates, compiles, imports or shells out; git is run by the operator (tree listing of the exact
    commit), and every file's git blob id is re-derived from its bytes before it is accepted;
  * external backtest/profitability statements are stored only as ``EXTERNAL_CLAIM`` metadata inside intake
    records; no validation, promotion, readiness or learning code reads intake records;
  * the only output is a research candidate: a ``StrategyDefinition`` built from an existing registered logic
    kind, handed to the existing research workflow with the intake id locked into its pre-registration.
"""

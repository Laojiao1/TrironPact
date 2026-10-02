import json
from pathlib import Path

from bench.e3.validate import source_fingerprint


ROOT = Path(__file__).resolve().parents[1]


def test_e3_reports_keep_e2_frozen_and_e4_out_of_scope():
    freeze = json.loads((ROOT / "bench/e2/freeze.json").read_text(encoding="utf-8"))
    refinement = json.loads((ROOT / "results/e3_refinement_audit.json").read_text(encoding="utf-8"))
    assert freeze["holdout_evaluated"] is True
    assert refinement["witness_classification"]["understrong"] == []
    assert refinement["fast_revocations"] == []
    assert len(source_fingerprint()) == 64


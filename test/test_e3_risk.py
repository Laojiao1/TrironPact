from bench.e3.risk import build
from bench.e3.specs import RISK_SPECS, risk_spec


def test_risk_specs_keep_layers_distinct_and_traceable():
    assert len(RISK_SPECS) == 4
    assert len({item.id for item in RISK_SPECS}) == len(RISK_SPECS)
    assert len({item.source for item in RISK_SPECS}) == 3
    assert sum(item.wrapper_defense is not None for item in RISK_SPECS) == 3
    assert {item.l3_status for item in RISK_SPECS if item.l3_url} == {"open_at_2026-10-02"}
    assert risk_spec("triton_add_x_stride").l3_url is None


def test_failed_workers_remain_unknown(monkeypatch):
    from bench.e3 import risk

    monkeypatch.setattr(risk, "_run", lambda case_id, seed, timeout: {"case_id": case_id, "status": "Unknown", "reason": "worker_timeout"})
    report = build()
    assert not report["go_risk"]
    assert report["counts"]["complete"] == 0
    assert report["counts"]["l1_witnesses"] == 0


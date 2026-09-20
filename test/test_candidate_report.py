"""第三阶段报告保留候选、来源、状态、拒绝和旧路径的分层证据。"""

import json

from pact.candidate_report import build_candidate_report, render_candidate_report


def _regression():
    return {"go_core": True, "runs": [
        {"detail": {"case": "A2", "layout": "padded", "mode": "dispatch", "path": "Fast"}},
        {"detail": {"case": "B", "layout": "offset", "mode": "dispatch", "path": "Fast"}},
        {"detail": {"case": "D", "layout": "x_offset", "mode": "dispatch", "path": "Fast"}},
    ]}


def test_candidate_report_has_six_cases_hint_pair_and_rejections():
    report = build_candidate_report(_regression())
    assert report["go_candidates"], report["checks"]
    assert len(report["cases"]) == 6 and len(report["hints"]) == 2
    assert report["totals"]["generated"] == sum(report["totals"][name] for name in ("statically_proven", "unproven", "unknown"))
    assert report["totals"]["generated"] > 0
    assert all(item["shape_status"] == item["span_status"] == "Unknown" for item in report["rejections"].values())
    for case in report["cases"].values():
        for group in ("shape_stride", "span"):
            assert case["candidates"][group]
            assert all(item["origin"]["source_line"] and item["binding"] and item["domain"] for item in case["candidates"][group])
    encoded = json.loads(json.dumps(report, ensure_ascii=False))
    assert encoded["go_candidates"] is True
    rendered = render_candidate_report(report)
    assert "逐条候选" in rendered and "旧 Guard 路径对照" in rendered and "拒绝边界" in rendered
    assert "新候选没有接入 Fast" in rendered
    assert "reject:wrong_mask" in rendered and "reject:wrong_stride" in rendered and "reject:vector_hint" in rendered

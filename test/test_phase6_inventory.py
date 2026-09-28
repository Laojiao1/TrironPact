"""检查第六阶段清单不把同源改名或 AST 支持冒充完整证明。"""

import json

import pytest

from bench.inventory import AUDIT_RECORDS, CATALOG, ROOT, build, function_source


def test_current_catalog_is_reproducibly_probeable():
    report = build()
    assert report["counts"]["registered"] == 15
    assert report["counts"]["syntax_supported"] == 14
    assert report["counts"]["syntax_unsupported"] == 1
    assert report["counts"]["pinned_external_sources"] == 2
    assert report["counts"]["supported_positive_by_family"] == {
        "elementwise_mapping": 9,
        "feature_broadcast": 2,
        "layout_2d": 3,
    }
    assert (report["counts"]["candidate_complete"], report["counts"]["guard_available"], report["counts"]["fast_feasible"]) == (14, 8, 8)
    assert all(row["audit"]["wrapper"] and row["audit"]["reference"] for row in report["entries"])
    assert all(len(row["sha256"]) == 64 for row in report["entries"])
    assert "未执行语义" in report["scope"]


def test_duplicate_body_rejected(tmp_path, monkeypatch):
    data = json.loads(CATALOG.read_text(encoding="utf-8"))
    extra = {**data["entries"][0], "id": "renamed_duplicate"}
    data["entries"].append(extra)
    monkeypatch.setitem(AUDIT_RECORDS, "renamed_duplicate", AUDIT_RECORDS["A"])
    target = tmp_path / "catalog.json"
    target.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="重复函数体"):
        build(target, ROOT)


def test_function_source_is_exact_body():
    source, line = function_source(ROOT / "scenarios/kernels.py", "masked_1d")
    assert line > 0 and "def masked_1d" in source
    assert "def run_fast" not in source

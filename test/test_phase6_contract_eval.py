"""有限真值标签与当前配方一一对应，重复策略不膨胀分母。"""

from bench.contract_eval import LABELS, RECIPES, recipe_key, summarize


def test_labeled_inputs_match_existing_isolated_recipes():
    assert set(recipe_key(item) for item in RECIPES) == set(LABELS)
    assert len(LABELS) == 20


def test_repair_policy_duplicate_counts_once():
    fake = {"checks": {"ok": True}, "go_guard": True,
            "runs": [{"recipe": {"name": "A", "layout": "transpose"}, "exit_code": 0,
                      "result": {"classification": "correct", "detail": {"path": "PyTorch Fallback", "direct_guard": {"status": "False"}}}},
                     {"recipe": {"name": "A", "layout": "transpose", "policy": "prefer_repair"}, "exit_code": 0,
                      "result": {"classification": "correct", "detail": {"path": "Relayout+Fast", "direct_guard": {"status": "False"}}}}]}
    result = summarize(fake)
    assert result["metrics"]["counts"]["total"] == 1
    assert result["metrics"]["counts"]["tn"] == 1
    assert result["combined_metrics"]["counts"]["total"] == 1 + len(result["benchmark_shadow_rows"])
    assert result["combined_metrics"]["counts"]["fp"] == 0
    assert {row["evidence_mode"] for row in result["benchmark_shadow_rows"]} == {"offline_shadow_candidate"}

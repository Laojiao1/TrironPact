import json
from dataclasses import asdict

import pytest

from pact.mutation import MutationRecipe, TensorRecipe, shadow_candidates, standard_recipes
from pact.mutation_oracle import classify_process


def test_recipes_are_bounded_and_reproducible():
    recipes = standard_recipes()
    assert {item.case for item in recipes} == {"A", "A2", "B", "D", "holdout_vector", "holdout_matrix", "pointer", "index"}
    assert len({item.id for item in recipes}) == len(recipes)
    assert all(MutationRecipe.from_dict(item.to_dict()) == item for item in recipes)
    assert {item.label for item in recipes if item.case == "B"} >= {"tile_minus", "tile_exact", "tile_plus", "offset", "stride_two"}
    with pytest.raises(ValueError):
        TensorRecipe((129,), (1,), 1, 129)
    with pytest.raises(ValueError):
        MutationRecipe.from_dict({**recipes[0].to_dict(), "execute": "bad"})


def test_shadow_target_changes_and_span_boundary():
    from pact.candidate_report import build_candidate_report

    report = build_candidate_report({"go_core": True, "runs": []})
    items = report["cases"]["A"]["candidates"]["shape_stride"]
    base = next(item for item in standard_recipes() if item.case == "A" and item.label == "contiguous")
    trans = next(item for item in standard_recipes() if item.case == "A" and item.label == "transpose")
    assert all(row["value"] is True for row in shadow_candidates(base, items))
    assert any(row["value"] is False for row in shadow_candidates(trans, items))


def test_classification_requires_matching_identity_and_explicit_result():
    recipe = standard_recipes()[0]
    payload = {"id": recipe.id, "fingerprint": recipe.fingerprint, "kind": "correct"}
    assert classify_process(recipe, 0, json.dumps(payload), "")["category"] == "correct"
    assert classify_process(recipe, 0, json.dumps({**payload, "kind": "metadata_mismatch"}), "")["category"] == "metadata_mismatch"
    assert classify_process(recipe, 2, json.dumps({**payload, "kind": "worker_exception", "stage": "execution", "message": "CUDA failure"}), "")["category"] == "inconclusive"
    assert classify_process(recipe, 2, json.dumps({**payload, "kind": "worker_exception", "stage": "execution", "message": "illegal memory access"}), "")["category"] == "memory_error_reported"
    assert classify_process(recipe, 2, json.dumps({**payload, "kind": "worker_exception", "stage": "execution", "type": "CompilationError"}), "")["category"] == "compile_or_launch_error"
    assert classify_process(recipe, 2, json.dumps({**payload, "kind": "worker_exception", "stage": "protocol"}), "")["category"] == "oracle_setup_error"
    assert classify_process(recipe, 0, json.dumps({**payload, "fingerprint": "bad"}), "")["category"] == "process_failure_unknown"
    assert classify_process(recipe, -9, "", "")["category"] == "process_failure_unknown"


def test_worker_rejects_invalid_json_without_gpu_execution():
    import subprocess
    import sys

    done = subprocess.run([sys.executable, "-m", "scenarios.mutation_worker"], input='{"version":1,"execute":"arbitrary"}', text=True, capture_output=True, timeout=15)
    result = json.loads(done.stdout)
    assert done.returncode == 2 and result["kind"] == "worker_exception" and result["stage"] == "protocol"


def test_oracle_timeout_remains_unknown(monkeypatch):
    import subprocess
    from pact import mutation_oracle

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 1)

    monkeypatch.setattr(mutation_oracle.subprocess, "run", timeout)
    assert mutation_oracle.run_mutation(standard_recipes()[0], timeout=1)["category"] == "timeout_unknown"

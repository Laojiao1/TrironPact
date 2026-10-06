from bench.e4.validate import _unexpected_worktree


def test_e4_validator_allows_only_e4_scope():
    status = " M README.md\n?? integration/\n?? bench/e4/\n?? test/test_e4_validate.py\n?? results/e4_regression.json"
    assert _unexpected_worktree(status) == []
    assert _unexpected_worktree(status + "\n M pact/e2_guard.py") == ["pact/e2_guard.py"]

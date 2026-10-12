from bench.e4.validate import _unexpected_worktree


def test_e4_validator_allows_only_e4_scope():
    status = (
        " M README.md\n M bench/e1/source.py\n M scripts/e1/setup_native_linux.sh\n"
        " M bench/e5/readiness.py\n M test/test_e1_corpus.py\n M test/test_e5_readiness.py\n"
        " M results/e1_semantic_audit.json\n?? integration/\n?? bench/e4/\n"
        "?? test/test_e4_validate.py\n?? results/e4_regression.json\n?? results/e4b_machine_4090.json"
    )
    assert _unexpected_worktree(status) == []
    assert _unexpected_worktree(status + "\n M pact/e2_guard.py") == ["pact/e2_guard.py"]

from bench.e4.correctness import _run


def test_failed_e4_worker_stays_unknown(monkeypatch):
    import subprocess

    def fail(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 2, stdout="", stderr="boom")

    monkeypatch.setattr(subprocess, "run", fail)
    row = _run(["python", "-m", "integration.worker", "transformer_layer_trace"])
    assert row["status"] == "Unknown"
    assert row["reason"] == "worker_failure"


def test_timeout_e4_worker_stays_unknown(monkeypatch):
    import subprocess

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 1)

    monkeypatch.setattr(subprocess, "run", timeout)
    row = _run(["python", "-m", "integration.worker", "inductor_custom_kernel_graph"])
    assert row["status"] == "Unknown"
    assert row["reason"] == "worker_timeout"

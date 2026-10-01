"""把冻结语料条目编译为 e2 离线 GuardPlan。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from bench.e2.contracts import _tensors
from bench.e2.coverage import PROJECT, _constexpr_values, _element_bytes, _kernel
from pact.e2_access import parse_extended_access_ir
from pact.e2_contracts import extract_bounded_contracts
from pact.e2_guard import E2GuardPlan, compile_e2_guard


def build_plan(entry: dict) -> E2GuardPlan:
    kernel = _kernel(entry)
    ir = parse_extended_access_ir(kernel, entry["pointers"], _element_bytes(entry), _constexpr_values(entry, kernel))
    tensors = _tensors(entry)
    bindings = tuple((item["parameter"], item["expression"]) for item in entry["smoke"]["bindings"])
    contracts = extract_bounded_contracts(ir, tensors, bindings, tuple(entry["smoke"]["grid"]), entry["reference_anchor"])
    paths = [
        PROJECT / "pact" / "e2_access.py",
        PROJECT / "pact" / "e2_contracts.py",
        PROJECT / "pact" / "e2_guard.py",
        PROJECT / "bench" / "e1" / entry["kernel_module"],
    ]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(PROJECT).as_posix().encode())
        digest.update(path.read_bytes())
    source_fingerprint = digest.hexdigest()
    if contracts.status != "Supported":
        return E2GuardPlan(entry["id"], tuple(tensors.values()), contracts.candidates, contracts.status, contracts.reason, source_fingerprint)
    return compile_e2_guard(entry["id"], tuple(tensors.values()), contracts.candidates, source_fingerprint)

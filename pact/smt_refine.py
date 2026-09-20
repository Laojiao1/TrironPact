"""受限 DSL 的 SMT 蕴含/冲突检查；只生成离线影子精简结果。"""

from __future__ import annotations

from dataclasses import dataclass

import z3

from pact.candidates import Candidate
from pact.contract_dsl import Condition, ValueRef


class CannotEncode(ValueError):
    pass


@dataclass(frozen=True)
class Domain:
    """已独立核对的 SMT 域；未列出的 symbol 不能偷偷变成自由变量。"""

    ranks: tuple[tuple[str, int], ...]
    scalars: tuple[str, ...] = ()
    facts: tuple[Condition, ...] = ()

    def __post_init__(self) -> None:
        if not self.ranks or len(dict(self.ranks)) != len(self.ranks) or any(not name or rank not in (1, 2) for name, rank in self.ranks) or len(set(self.scalars)) != len(self.scalars):
            raise ValueError("SMT 域的 Tensor 维度或标量无效")


class Encoder:
    def __init__(self, domain: Domain):
        self.domain = domain
        self.ranks = dict(domain.ranks)
        self.vars: dict[tuple, z3.ArithRef] = {}

    def variable(self, kind: str, name: str, axis: int | None = None):
        key = (kind, name, axis)
        if key not in self.vars:
            self.vars[key] = z3.Int(f"{kind}:{name}:{axis if axis is not None else '-'}")
        return self.vars[key]

    def value(self, ref: ValueRef):
        if ref.kind == "constant":
            return z3.IntVal(ref.value)
        if ref.kind == "scalar":
            if ref.scalar not in self.domain.scalars:
                raise CannotEncode("未声明标量")
            return self.variable("scalar", ref.scalar)
        if ref.tensor not in self.ranks:
            raise CannotEncode("未声明 Tensor")
        if ref.kind in ("size", "stride"):
            if ref.axis is None or ref.axis >= self.ranks[ref.tensor]:
                raise CannotEncode("Tensor 维度超出支持域")
            return self.variable(ref.kind, ref.tensor, ref.axis)
        if ref.kind in ("effective_ptr", "storage_offset", "element_bytes", "storage_nbytes"):
            return self.variable(ref.kind, ref.tensor)
        raise CannotEncode("未支持属性")

    def condition(self, cond: Condition):
        if cond.kind == "eq":
            return self.value(cond.left) == self.value(cond.right)
        if cond.kind == "mod_eq":
            return self.value(cond.left) % cond.divisor == cond.remainder
        if cond.kind == "and":
            return z3.And(*(self.condition(part) for part in cond.parts))
        if cond.kind == "or":
            return z3.Or(*(self.condition(part) for part in cond.parts))
        raise CannotEncode(f"未完整编码的条件：{cond.kind}")

    def domain_constraints(self):
        constraints = []
        for name, rank in self.ranks.items():
            for axis in range(rank):
                constraints.extend((self.variable("size", name, axis) >= 1, self.variable("stride", name, axis) >= 0))
            constraints.extend((self.variable("storage_offset", name) >= 0, self.variable("element_bytes", name) >= 1, self.variable("storage_nbytes", name) >= 0, self.variable("effective_ptr", name) >= 0))
        constraints.extend(self.variable("scalar", name) >= 0 for name in self.domain.scalars)
        constraints.extend(self.condition(item) for item in self.domain.facts)
        return constraints


def _check(terms: list, timeout_ms: int) -> tuple[str, str, dict]:
    solver = z3.Solver()
    solver.set(timeout=timeout_ms)
    solver.add(*terms)
    status = solver.check()
    model = {}
    if status == z3.sat:
        for declaration in solver.model().decls():
            value = solver.model()[declaration]
            if z3.is_int_value(value):
                model[declaration.name()] = value.as_long()
    return str(status), solver.sexpr(), model


def check_redundancy(candidates: list[Candidate], domain: Domain, *, timeout_ms: int = 2000) -> dict:
    if timeout_ms < 1 or timeout_ms > 30000:
        raise ValueError("求解超时预算无效")
    encoder = Encoder(domain)
    try:
        base = encoder.domain_constraints()
    except CannotEncode as exc:
        return {"status": "Unknown", "reason": str(exc), "kept": list(range(len(candidates))), "removed": [], "queries": []}
    domain_status, domain_query, _ = _check(base, timeout_ms)
    queries = [{"kind": "domain", "status": domain_status, "smt2": domain_query}]
    if domain_status != "sat":
        return {"status": "Conflict" if domain_status == "unsat" else "Unknown", "reason": "域不可满足或求解未知", "kept": list(range(len(candidates))), "removed": [], "queries": queries}
    kept = list(range(len(candidates)))
    removed = []
    groups: dict[tuple, list[int]] = {}
    for index, candidate in enumerate(candidates):
        if candidate.condition is not None and candidate.evidence == "Statically-Proven":
            groups.setdefault((candidate.purpose, candidate.domain), []).append(index)
    for group_key, indexes in groups.items():
        if len(indexes) < 2:
            continue
        try:
            encoded = {index: encoder.condition(candidates[index].condition) for index in indexes}
        except CannotEncode as exc:
            queries.append({"kind": "group", "group": str(group_key), "status": "unknown", "reason": str(exc)})
            continue
        status, query, _ = _check(base + list(encoded.values()), timeout_ms)
        queries.append({"kind": "consistency", "group": str(group_key), "status": status, "smt2": query})
        if status != "sat":
            continue
        active = indexes[:]
        for index in reversed(indexes):
            others = [encoded[item] for item in active if item != index]
            if not others:
                continue
            status, query, model = _check(base + others + [z3.Not(encoded[index])], timeout_ms)
            queries.append({"kind": "redundancy", "candidate": index, "status": status, "smt2": query, "model": model})
            if status == "unsat":
                active.remove(index)
                kept.remove(index)
                removed.append(index)
    return {"status": "Supported", "reason": "仅同目的、同域且完整编码的静态条件可删除", "kept": sorted(kept), "removed": sorted(removed), "queries": queries}

"""建立/读取固定版 e1 语料；所有原始资产均通过 SHA-256 绑定。"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from bench.e1.acquire import ROOT
from bench.e1.selection import selected
from bench.e1.source import body_fingerprint, extract, features

CATALOG = ROOT / "catalog.json"
HEADER = "import torch\nimport triton\nimport triton.language as tl\nfrom triton.language.extra.cuda.libdevice import tanh, rsqrt\ndebug = False\n_CASTING_MODE_NONE: tl.constexpr = tl.constexpr(-1)\n_CASTING_MODE_LLAMA: tl.constexpr = tl.constexpr(0)\n_CASTING_MODE_GEMMA: tl.constexpr = tl.constexpr(1)\n"


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def build() -> dict:
    """从已下载资产生成审计清单，不执行任何 Kernel 或候选规则。"""
    sources = json.loads((ROOT / "sources.json").read_text())
    rows = []
    for index, item in enumerate(selected()):
        source = sources[item.source]
        path = ROOT / "upstream" / item.source / item.file
        function = extract(path, item.function)
        pointers = {a.arg: a.arg for a in function.node.args.args if a.annotation is None or not ast.unparse(a.annotation).endswith("constexpr")}
        # 摸底并不推断参数类型；正向通过独立配方只登记真实 Tensor 指针。
        if item.smoke:
            pointers = {b.parameter: b.expression for b in item.smoke.bindings if b.expression in {t.name for t in item.smoke.tensors}}
        else:
            # 挑战原始签名的指针名经人工初筛，仅用于暴露首个拒绝节点。
            pointers = {name: name for name in pointers if "ptr" in name.lower() or name in {"X", "Y", "Z", "In", "Out", "e", "g", "h", "Q", "K", "cos", "sin", "input", "out"}}
        role = "challenge" if item.smoke is None else ("holdout" if index % 4 == 2 else "development")
        module = HEADER
        if item.helper:
            module += "\n" + extract(path, item.helper).source + "\n"
        module += "\n" + function.source + "\n"
        module_path = ROOT / "kernels" / f"{item.id}.py"
        # 只为有独立参考的样本生成加载封套；函数和 helper 逐字提取。
        if item.smoke:
            module_path.parent.mkdir(exist_ok=True)
            module_path.write_text(module, encoding="utf-8")
        row = {**asdict(item), "split": role, "revision": source["revision"],
               "repository": source["repository"], "license_id": source["license_id"],
               "license_sha256": source["assets"]["LICENSE"]["sha256"],
               "source_url": f"https://github.com/{source['repository']}/blob/{source['revision']}/{item.file}#L{function.line}",
               "line": function.line, "file_sha256": sha256(path.read_bytes()),
               "function_sha256": sha256(function.source.encode()), "normalized_body_sha256": body_fingerprint(function.source),
               "signature": [a.arg for a in function.node.args.args], "pointers": pointers,
               "features": features(function.source), "kernel_module": str(module_path.relative_to(ROOT)).replace("\\", "/") if item.smoke else None,
               "kernel_module_sha256": sha256(module.encode()) if item.smoke else None,
               "adaptation": "仅移除外层作用域缩进并补充导入/固定 helper 常量；函数体和 decorator 未修改" if item.smoke else "未适配、未执行",
               "reference_file_sha256": sha256((ROOT / "upstream" / item.source / item.test_file).read_bytes()) if item.test_file else None,
               "semantic_status": "declared_independent_reference" if item.smoke else "not_required_challenge",
               "candidate_status": "Unknown", "guard_status": "not_registered", "fast_status": "not_eligible",
               "alias_policy": "独立新分配输入/输出，禁止共享 storage；不证明一般 alias" if item.smoke else "Unknown",
               "input_recipe": "CUDA；非空连续新分配；float32/int32/bool；固定两种 seed=17,42；smoke shapes 见 tensors；部分 shape 为有界适配" if item.smoke else None,
               "tolerance_provenance": "上游精度域/参考公式固定；缓存统计量独立核对；未放宽失败容差" if item.smoke else None}
        rows.append(row)
    normalized = [row["normalized_body_sha256"] for row in rows]
    if len(normalized) != len(set(normalized)):
        raise ValueError("改名归一化后发现重复函数体")
    return {"version": 1, "freeze_status": "pending_semantic_smoke", "split_policy": "选型顺序每四个的第三个进入 holdout；挑战另列；e1 允许真值/来源 smoke，禁止据此调分析规则", "entries": rows}


def load(path: Path = CATALOG) -> dict:
    """读取清单并拒绝来源、函数、参考或加载封套指纹漂移。"""
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("version") != 1 or not isinstance(data.get("entries"), list):
        raise ValueError("语料 schema 版本错误")
    ids, bodies = set(), set()
    sources = json.loads((ROOT / "sources.json").read_text())
    for key, source in sources.items():
        if "LICENSE" not in source["assets"]:
            raise ValueError(f"缺少固定许可：{key}")
        for asset, info in source["assets"].items():
            path = (ROOT / "upstream" / key / asset).resolve()
            if not path.is_relative_to((ROOT / "upstream").resolve()) or sha256(path.read_bytes()) != info["sha256"]:
                raise ValueError(f"原始资产指纹失配：{key}/{asset}")
    for row in data["entries"]:
        if row["id"] in ids or row["normalized_body_sha256"] in bodies:
            raise ValueError("语料 ID 或函数体重复")
        ids.add(row["id"]); bodies.add(row["normalized_body_sha256"])
        source = sources[row["source"]]
        if source["revision"] != row["revision"] or len(row["revision"]) != 40:
            raise ValueError("来源提交漂移或未固定")
        if source.get("license_id") != row.get("license_id") or source["assets"]["LICENSE"]["sha256"] != row.get("license_sha256"):
            raise ValueError(f"许可证元数据失配：{row['id']}")
        for field, hash_field in [("file", "file_sha256"), ("test_file", "reference_file_sha256")]:
            if row[field]:
                target = (ROOT / "upstream" / row["source"] / row[field]).resolve()
                if not target.is_relative_to((ROOT / "upstream").resolve()) or sha256(target.read_bytes()) != row[hash_field]:
                    raise ValueError(f"来源/参考指纹失配：{row['id']} {field}")
        fn = extract(ROOT / "upstream" / row["source"] / row["file"], row["function"])
        if sha256(fn.source.encode()) != row["function_sha256"] or body_fingerprint(fn.source) != row["normalized_body_sha256"]:
            raise ValueError(f"函数指纹失配：{row['id']}")
        if row["smoke"] is not None:
            bindings = row["smoke"]["bindings"]
            if len(bindings) != len(row["signature"]) or [b["parameter"] for b in bindings] != row["signature"]:
                raise ValueError(f"完整签名绑定失配：{row['id']}")
            if row["split"] not in {"development", "holdout"} or not row["reference_anchor"] or not row["test_file"]:
                raise ValueError("正向样本缺少独立参考/划分")
            path = (ROOT / row["kernel_module"]).resolve()
            if not path.is_relative_to(ROOT.resolve()) or sha256(path.read_bytes()) != row["kernel_module_sha256"]:
                raise ValueError(f"加载封套指纹漂移：{row['id']}")
            compiled_source = extract(path, fn.node.name).source
            if compiled_source != fn.source:
                raise ValueError(f"原始函数被改写：{row['id']}")
        elif row["split"] != "challenge":
            raise ValueError("无参考样本只能进入挑战集")
    return data


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true", help="仅冻结前允许重建")
    args = parser.parse_args()
    if args.build:
        if CATALOG.exists() and json.loads(CATALOG.read_text()).get("freeze_status") == "frozen":
            raise ValueError("已冻结语料不能隐式重建；需显式版本迁移")
        CATALOG.write_text(json.dumps(build(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    data = load()
    print(f"registered={len(data['entries'])} source_binding_hash_audit=passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

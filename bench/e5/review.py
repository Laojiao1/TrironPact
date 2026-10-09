"""生成冻结的第二审阅包，并严格核验独立审阅记录。"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT / "results"
PACKAGE = PROJECT / "artifact" / "e5_review"
CATALOG = PROJECT / "bench" / "e1" / "catalog.json"
RISK_REPORT = RESULTS / "e3_real_risk_cases.json"
MANIFEST = PACKAGE / "review_manifest.json"
TEMPLATE = PACKAGE / "review_template.json"

SELECTED_KERNEL_IDS = (
    "triton_add",
    "pytorch_double_strided",
    "liger_swiglu",
    "flag_slice",
    "liger_softmax",
    "unsloth_layernorm",
    "unsloth_rmsnorm",
)

KERNEL_STATUS = {"confirmed", "rejected", "unknown"}
ACCESS_LABELS = {"Supported", "Unknown", "Unsupported"}
OBSERVED_BEHAVIORS = {"numeric_mismatch", "correct", "exception", "timeout", "unknown"}
GUARD_BEHAVIORS = {"blocked", "allowed", "unknown"}
UPSTREAM_CLAIMS = {"confirmed", "rejected", "not_claimed", "unknown"}
FINAL_STATUS = {"confirmed", "rejected", "unknown"}


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=PROJECT, capture_output=True, text=True, timeout=30, check=True).stdout.rstrip("\r\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _relative(path: Path) -> str:
    return path.relative_to(PROJECT).as_posix()


def _git_blob(commit: str, relative: str) -> bytes:
    done = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=PROJECT,
        capture_output=True,
        timeout=30,
        check=True,
    )
    return done.stdout


def _risk_rows(report: dict) -> list[dict]:
    rows = []
    for row in report["rows"]:
        if row.get("status") != "complete":
            continue
        case = row["detail"]["case"]
        rows.append(
            {
                "case_id": row["case_id"],
                "kernel_id": case["kernel_id"],
                "source": case["source"],
                "worker_command": ["python", "-m", "bench.e3.worker", row["case_id"]],
            }
        )
    return rows


def build_manifest(freeze_commit: str) -> dict:
    """建立盲审输入清单；不写入项目的语义、Guard 或风险结论。"""

    resolved = _git("rev-parse", freeze_commit)
    catalog = json.loads(_git_blob(resolved, _relative(CATALOG)).decode("utf-8"))
    risk_raw = _git_blob(resolved, _relative(RISK_REPORT))
    risk_report = json.loads(risk_raw.decode("utf-8"))
    entries = {row["id"]: row for row in catalog["entries"]}
    rows = []
    for kernel_id in SELECTED_KERNEL_IDS:
        row = entries[kernel_id]
        source = PROJECT / "bench" / "e1" / "upstream" / row["source"] / row["file"]
        reference = PROJECT / "bench" / "e1" / "upstream" / row["source"] / row["test_file"]
        rows.append(
            {
                "kernel_id": kernel_id,
                "source": row["source"],
                "family": row["family"],
                "split": row["split"],
                "upstream_revision": row["revision"],
                "source_file": _relative(source),
                "source_file_sha256": hashlib.sha256(_git_blob(resolved, _relative(source))).hexdigest(),
                "function": row["function"],
                "reference_file": _relative(reference),
                "reference_file_sha256": hashlib.sha256(_git_blob(resolved, _relative(reference))).hexdigest(),
                "reference_anchor": row["reference_anchor"],
                "catalog_entry_sha256": _json_sha256(row),
                "catalog_path": _relative(CATALOG),
            }
        )
    risks = _risk_rows(risk_report)
    return {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "freeze_commit": resolved,
        "formal_kernel_denominator": 28,
        "selection_policy": "预先固定 7/28；覆盖五个来源、四类结构、development/holdout，并包含全部风险所涉及的三个不同 Kernel。",
        "blind_review_policy": "初始判断前不得查看 results/e1_semantic_audit、e2_*coverage、e3_real_risk_cases 或 e5 readiness 中的项目结论。",
        "kernels": rows,
        "risk_cases": risks,
        "risk_source_sha256": hashlib.sha256(risk_raw).hexdigest(),
    }


def build_template(manifest: dict) -> dict:
    kernel_records = []
    for row in manifest["kernels"]:
        kernel_records.append(
            {
                "kernel_id": row["kernel_id"],
                "initial_assessment": {
                    "semantic_correspondence": None,
                    "wrapper_binding": None,
                    "reference_truth": None,
                    "access_label": None,
                    "evidence": [],
                    "notes": "",
                    "completed_at": None,
                    "frozen_before_reconciliation": False,
                },
                "reconciliation": {
                    "project_label_revealed_after_initial": False,
                    "agrees_with_project": None,
                    "final_status": None,
                    "resolution_status": None,
                    "resolution_evidence": [],
                },
            }
        )
    risk_records = []
    for row in manifest["risk_cases"]:
        risk_records.append(
            {
                "case_id": row["case_id"],
                "initial_assessment": {
                    "observed_behavior": None,
                    "reference_truth": None,
                    "guard_behavior": None,
                    "risk_levels": [],
                    "upstream_vulnerability_claim": None,
                    "evidence": [],
                    "notes": "",
                    "completed_at": None,
                    "frozen_before_reconciliation": False,
                },
                "reconciliation": {
                    "project_label_revealed_after_initial": False,
                    "agrees_with_project": None,
                    "final_status": None,
                    "resolution_status": None,
                    "resolution_evidence": [],
                },
            }
        )
    return {
        "schema_version": 1,
        "status": "template_unfilled",
        "frozen_git_head": manifest["freeze_commit"],
        "review_manifest_sha256": _json_sha256(manifest),
        "reviewer": {
            "reviewer_id": "",
            "independent": None,
            "no_prior_rule_or_label_involvement": None,
            "independence_declaration": "",
            "started_at": None,
            "completed_at": None,
        },
        "formal_kernel_denominator": 28,
        "reviewed_kernel_count": 0,
        "reviewed_risk_case_ids": [],
        "review_records": kernel_records,
        "risk_review_records": risk_records,
        "disagreements": [],
        "disagreements_resolved": False,
    }


@dataclass(frozen=True)
class ReviewValidation:
    complete: bool
    errors: tuple[str, ...]
    kernel_ids: tuple[str, ...]
    risk_ids: tuple[str, ...]


def validate_manifest(manifest: object) -> tuple[str, ...]:
    """从冻结 Git object 重算清单，工作树变化不能替代被审版本。"""

    if not isinstance(manifest, dict):
        return ("审阅清单根对象必须为 JSON object",)
    errors: list[str] = []
    commit = manifest.get("freeze_commit")
    if not isinstance(commit, str):
        return ("freeze_commit 缺失",)
    try:
        resolved = _git("rev-parse", commit)
        catalog = json.loads(_git_blob(resolved, _relative(CATALOG)).decode("utf-8"))
        risk_raw = _git_blob(resolved, _relative(RISK_REPORT))
    except (subprocess.CalledProcessError, json.JSONDecodeError, UnicodeDecodeError):
        return ("freeze_commit 或冻结资产无法读取",)
    if resolved != commit:
        errors.append("freeze_commit 必须使用完整提交哈希")
    entries = {row["id"]: row for row in catalog["entries"]}
    kernels = manifest.get("kernels")
    kernels = kernels if isinstance(kernels, list) else []
    ids = [row.get("kernel_id") for row in kernels if isinstance(row, dict)]
    if tuple(ids) != SELECTED_KERNEL_IDS or len(ids) != len(set(ids)):
        errors.append("冻结 Kernel 顺序、集合或唯一性失配")
    for row in kernels:
        if not isinstance(row, dict) or row.get("kernel_id") not in entries:
            errors.append("冻结 Kernel 记录非法")
            continue
        entry = entries[row["kernel_id"]]
        source_relative = f"bench/e1/upstream/{entry['source']}/{entry['file']}"
        reference_relative = f"bench/e1/upstream/{entry['source']}/{entry['test_file']}"
        expected = {
            "source": entry["source"],
            "family": entry["family"],
            "split": entry["split"],
            "upstream_revision": entry["revision"],
            "source_file": source_relative,
            "source_file_sha256": hashlib.sha256(_git_blob(resolved, source_relative)).hexdigest(),
            "function": entry["function"],
            "reference_file": reference_relative,
            "reference_file_sha256": hashlib.sha256(_git_blob(resolved, reference_relative)).hexdigest(),
            "reference_anchor": entry["reference_anchor"],
            "catalog_entry_sha256": _json_sha256(entry),
            "catalog_path": _relative(CATALOG),
        }
        if any(row.get(key) != value for key, value in expected.items()):
            errors.append(f"{row['kernel_id']}: 冻结来源或哈希失配")
    try:
        risk_report = json.loads(risk_raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        errors.append("冻结风险报告无法解析")
    else:
        if manifest.get("risk_cases") != _risk_rows(risk_report):
            errors.append("冻结风险案例集合失配")
    if manifest.get("risk_source_sha256") != hashlib.sha256(risk_raw).hexdigest():
        errors.append("冻结风险报告哈希失配")
    if manifest.get("formal_kernel_denominator") != 28:
        errors.append("正式 Kernel 分母不是 28")
    return tuple(errors)


def _valid_time(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        return datetime.fromisoformat(value).tzinfo is not None
    except ValueError:
        return False


def _parse_time(value: object) -> datetime | None:
    if not _valid_time(value):
        return None
    return datetime.fromisoformat(str(value))


def _validate_reconciliation(record_id: str, value: object, errors: list[str]) -> bool:
    if not isinstance(value, dict):
        errors.append(f"{record_id}: reconciliation 缺失")
        return False
    agreement = value.get("agrees_with_project")
    if value.get("project_label_revealed_after_initial") is not True:
        errors.append(f"{record_id}: 未证明先冻结初始判断再揭示项目标签")
    if type(agreement) is not bool:
        errors.append(f"{record_id}: agrees_with_project 必须为 bool")
        return False
    expected_resolution = "not_needed" if agreement else "resolved"
    if value.get("resolution_status") != expected_resolution:
        errors.append(f"{record_id}: resolution_status 应为 {expected_resolution}")
    if value.get("final_status") not in FINAL_STATUS:
        errors.append(f"{record_id}: final_status 非法")
    if not agreement and not value.get("resolution_evidence"):
        errors.append(f"{record_id}: 分歧缺少 resolution_evidence")
    return not agreement


def validate_review(data: object, manifest: dict) -> ReviewValidation:
    """严格验证完整审阅；模板、重复计数和未解决分歧均 fail closed。"""

    errors: list[str] = list(validate_manifest(manifest))
    if not isinstance(data, dict):
        return ReviewValidation(False, ("根对象必须为 JSON object",), (), ())
    if data.get("schema_version") != 1 or data.get("status") != "complete":
        errors.append("schema_version/status 不表示完整审阅")
    if data.get("frozen_git_head") != manifest.get("freeze_commit"):
        errors.append("frozen_git_head 与审阅清单不一致")
    if data.get("review_manifest_sha256") != _json_sha256(manifest):
        errors.append("review_manifest_sha256 失配")
    reviewer = data.get("reviewer")
    if not isinstance(reviewer, dict):
        errors.append("reviewer 缺失")
    else:
        if not reviewer.get("reviewer_id"):
            errors.append("reviewer_id 为空")
        if reviewer.get("independent") is not True or reviewer.get("no_prior_rule_or_label_involvement") is not True:
            errors.append("审阅独立性声明不完整")
        if len(str(reviewer.get("independence_declaration", "")).strip()) < 20:
            errors.append("independence_declaration 过短")
        if not _valid_time(reviewer.get("started_at")) or not _valid_time(reviewer.get("completed_at")):
            errors.append("审阅起止时间缺失或无时区")
        else:
            started = _parse_time(reviewer["started_at"])
            completed = _parse_time(reviewer["completed_at"])
            if started is not None and completed is not None and started > completed:
                errors.append("审阅 completed_at 早于 started_at")
    expected_kernels = {row["kernel_id"] for row in manifest["kernels"]}
    records = data.get("review_records")
    records = records if isinstance(records, list) else []
    kernel_ids = tuple(row.get("kernel_id") for row in records if isinstance(row, dict) and isinstance(row.get("kernel_id"), str))
    if len(kernel_ids) != len(set(kernel_ids)):
        errors.append("review_records 存在重复 kernel_id")
    if set(kernel_ids) != expected_kernels or len(kernel_ids) != 7:
        errors.append("review_records 未精确覆盖冻结的 7 个 Kernel")
    disagreement_ids: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            errors.append("Kernel 审阅记录不是 object")
            continue
        kernel_id = str(record.get("kernel_id", "Unknown"))
        initial = record.get("initial_assessment")
        if not isinstance(initial, dict):
            errors.append(f"{kernel_id}: initial_assessment 缺失")
            continue
        for field in ("semantic_correspondence", "wrapper_binding", "reference_truth"):
            if initial.get(field) not in KERNEL_STATUS:
                errors.append(f"{kernel_id}: {field} 非法")
        if initial.get("access_label") not in ACCESS_LABELS:
            errors.append(f"{kernel_id}: access_label 非法")
        if len(initial.get("evidence", ())) < 2:
            errors.append(f"{kernel_id}: 初始判断至少需要两条证据")
        if initial.get("frozen_before_reconciliation") is not True or not _valid_time(initial.get("completed_at")):
            errors.append(f"{kernel_id}: 初始判断未按时冻结")
        if _validate_reconciliation(kernel_id, record.get("reconciliation"), errors):
            disagreement_ids.add(f"kernel:{kernel_id}")
    expected_risks = {row["case_id"] for row in manifest["risk_cases"]}
    risk_records = data.get("risk_review_records")
    risk_records = risk_records if isinstance(risk_records, list) else []
    risk_ids = tuple(row.get("case_id") for row in risk_records if isinstance(row, dict) and isinstance(row.get("case_id"), str))
    if len(risk_ids) != len(set(risk_ids)):
        errors.append("risk_review_records 存在重复 case_id")
    if set(risk_ids) != expected_risks:
        errors.append("risk_review_records 未精确覆盖全部冻结风险案例")
    for record in risk_records:
        if not isinstance(record, dict):
            errors.append("风险审阅记录不是 object")
            continue
        case_id = str(record.get("case_id", "Unknown"))
        initial = record.get("initial_assessment")
        if not isinstance(initial, dict):
            errors.append(f"{case_id}: initial_assessment 缺失")
            continue
        if initial.get("observed_behavior") not in OBSERVED_BEHAVIORS:
            errors.append(f"{case_id}: observed_behavior 非法")
        if initial.get("reference_truth") not in KERNEL_STATUS:
            errors.append(f"{case_id}: reference_truth 非法")
        if initial.get("guard_behavior") not in GUARD_BEHAVIORS:
            errors.append(f"{case_id}: guard_behavior 非法")
        levels = initial.get("risk_levels")
        if not isinstance(levels, list) or not levels or not set(levels) <= {"L1", "L2", "L3"}:
            errors.append(f"{case_id}: risk_levels 非法或为空")
        if initial.get("upstream_vulnerability_claim") not in UPSTREAM_CLAIMS:
            errors.append(f"{case_id}: upstream_vulnerability_claim 非法")
        if len(initial.get("evidence", ())) < 2:
            errors.append(f"{case_id}: 初始判断至少需要两条证据")
        if initial.get("frozen_before_reconciliation") is not True or not _valid_time(initial.get("completed_at")):
            errors.append(f"{case_id}: 初始判断未按时冻结")
        if _validate_reconciliation(case_id, record.get("reconciliation"), errors):
            disagreement_ids.add(f"risk:{case_id}")
    disagreements = data.get("disagreements")
    disagreements = disagreements if isinstance(disagreements, list) else []
    recorded_ids = set()
    for row in disagreements:
        if not isinstance(row, dict):
            errors.append("分歧记录不是 object")
            continue
        record_id = row.get("record_id")
        record_type = row.get("record_type")
        key = f"{record_type}:{record_id}"
        recorded_ids.add(key)
        if record_type not in {"kernel", "risk"} or not row.get("field") or row.get("resolved") is not True:
            errors.append(f"{key}: 分歧字段或 resolved 非法")
        if not row.get("initial_value") or not row.get("project_value") or not row.get("resolution") or not row.get("evidence"):
            errors.append(f"{key}: 分歧内容不完整")
    if len(recorded_ids) != len(disagreements):
        errors.append("disagreements 存在重复 record_type/record_id")
    if recorded_ids != disagreement_ids:
        errors.append("disagreements 未与所有 record-level 分歧一一对应")
    if data.get("disagreements_resolved") is not True:
        errors.append("disagreements_resolved 不是 true")
    if data.get("formal_kernel_denominator") != 28 or data.get("reviewed_kernel_count") != len(expected_kernels):
        errors.append("正式分母或 reviewed_kernel_count 不正确")
    if set(data.get("reviewed_risk_case_ids", ())) != expected_risks:
        errors.append("reviewed_risk_case_ids 不完整")
    return ReviewValidation(not errors, tuple(errors), tuple(sorted(set(kernel_ids))), tuple(sorted(set(risk_ids))))


def validate_completed_review(path: Path | None = None, manifest_path: Path | None = None) -> ReviewValidation:
    path = path or RESULTS / "e5_second_review.json"
    manifest_path = manifest_path or MANIFEST
    if not path.is_file() or not manifest_path.is_file():
        return ReviewValidation(False, ("审阅结果或冻结清单缺失",), (), ())
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        return ReviewValidation(False, (f"JSON 无法解析：{error}",), (), ())
    return validate_review(data, manifest)


def write_package(freeze_commit: str) -> tuple[dict, dict]:
    PACKAGE.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(freeze_commit)
    template = build_template(manifest)
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    TEMPLATE.write_text(json.dumps(template, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    validation = validate_manifest(manifest)
    report = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "status": "ready_for_independent_reviewer" if not validation else "invalid",
        "review_complete": False,
        "freeze_commit": manifest["freeze_commit"],
        "manifest_sha256": _json_sha256(manifest),
        "template_sha256": _sha256(TEMPLATE),
        "counts": {"formal_kernel_denominator": 28, "selected_kernels": len(manifest["kernels"]), "risk_cases": len(manifest["risk_cases"])},
        "checks": {
            "manifest_matches_frozen_git_blobs": not validation,
            "five_sources": len({row["source"] for row in manifest["kernels"]}) == 5,
            "four_families": len({row["family"] for row in manifest["kernels"]}) == 4,
            "development_and_holdout": {row["split"] for row in manifest["kernels"]} == {"development", "holdout"},
            "all_risk_cases_registered": len(manifest["risk_cases"]) == 4,
            "template_intentionally_incomplete": not validate_review(template, manifest).complete,
        },
        "validation_errors": validation,
        "scope": "只表示审阅包可交付；第二审阅人尚未填写，不能满足 readiness 条件 7。",
    }
    (RESULTS / "e5_second_review_package.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# e5 第二审阅人冻结包",
        "",
        f"- 状态：`{report['status']}`；冻结提交：`{report['freeze_commit']}`。",
        f"- 计数：{report['counts']}。",
        f"- 边界：{report['scope']}",
        "",
        *[f"- [{'x' if value else ' '}] {key}" for key, value in report["checks"].items()],
        "",
        "交付文件：`artifact/e5_review/review_manifest.json`、`review_template.json`、`review_guide.md`。",
        "",
    ]
    (RESULTS / "e5_second_review_package.md").write_text("\n".join(lines), encoding="utf-8")
    return manifest, template


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    package = subparsers.add_parser("package")
    package.add_argument("--freeze-commit", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--input", type=Path, default=RESULTS / "e5_second_review.json")
    args = parser.parse_args()
    if args.command == "package":
        manifest, _ = write_package(args.freeze_commit)
        print(json.dumps({"freeze_commit": manifest["freeze_commit"], "kernels": len(manifest["kernels"]), "risk_cases": len(manifest["risk_cases"])}, ensure_ascii=False))
        return 0
    result = validate_completed_review(args.input)
    print(json.dumps({"complete": result.complete, "errors": result.errors, "kernel_ids": result.kernel_ids, "risk_ids": result.risk_ids}, ensure_ascii=False))
    return 0 if result.complete else 2


if __name__ == "__main__":
    raise SystemExit(main())


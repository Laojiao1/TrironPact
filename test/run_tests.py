"""一键运行阶段测试和隔离回归，并打印简短验收结果。"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="一键验证 TritonPact 契约分析与分派")
    parser.add_argument("--quick", action="store_true", help="只运行一轮案例；默认关键案例各运行十次")
    parser.add_argument("--output", type=Path, help="将本次隔离回归另存到指定 Markdown 路径")
    args = parser.parse_args()

    print("[1/2] 运行测试", flush=True)
    tests = subprocess.run([sys.executable, "-m", "pytest", "-q", "test/test_contract_dsl.py", "test/test_access_ir.py", "test/test_stage_integration.py", "test/test_poc.py", "test/test_candidates.py", "test/test_shape_stride.py", "test/test_alignment.py", "test/test_span.py", "test/test_holdouts.py", "test/test_candidate_report.py", "test/test_mutation.py", "test/test_smt_refine.py", "test/test_guard_dispatch.py"], cwd=ROOT, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(tests.stdout, end="", flush=True)
    if tests.returncode:
        return tests.returncode

    print("[2/2] 运行隔离案例", flush=True)
    repeats = "1" if args.quick else "10"
    result_path = args.output or ROOT / "results" / ("dispatch_regression_quick.md" if args.quick else "dispatch_regression.md")
    demo = subprocess.run([sys.executable, "main.py", "demo", "--repeats", repeats, "--output", str(result_path)], cwd=ROOT, check=False)
    data_path = result_path.with_suffix(".json")
    if not result_path.exists() or not data_path.exists():
        print("没有生成结果文件")
        return demo.returncode or 1

    data = json.loads(data_path.read_text(encoding="utf-8"))
    stage5_ok = True
    if result_path.stem.startswith("guard_regression"):
        from pact.cost_model import COST_VERSION, CostTable
        from pact.guard_plan import SPECS, compile_guard_plan

        summary = re.search(r"\d+ passed(?:, \d+ skipped)? in [^\n]+", tests.stdout)
        guard_path = ROOT / "results/guard_dispatch_report.json"
        cost_path = ROOT / "results/cost_calibration.json"
        guard_data = json.loads(guard_path.read_text(encoding="utf-8")) if guard_path.exists() else {}
        cost_data = json.loads(cost_path.read_text(encoding="utf-8")) if cost_path.exists() else {}
        current_plans = {name: compile_guard_plan(name) for name in SPECS}
        guard_current = guard_data.get("go_guard") is True and set(guard_data.get("plans", {})) == set(current_plans) and all(
            guard_data["plans"][name]["fingerprint"] == plan.fingerprint and plan.status == "Supported"
            for name, plan in current_plans.items())
        loaded_cost = CostTable.from_report(cost_path)
        cost_current = (loaded_cost.version == COST_VERSION and len(loaded_cost.entries) == len(cost_data.get("cases", ()))
                        and len(loaded_cost.entries) > 0 and all(
                            entry.key[0] in current_plans and entry.fingerprint == current_plans[entry.key[0]].fingerprint
                            for entry in loaded_cost.entries))
        validation = {"pytest_exit_code": tests.returncode, "pytest_summary": summary.group(0) if summary else "Unknown",
                      "legacy_isolation_checks": data["checks"], "legacy_isolation_passed": data["go_core"],
                      "guard_report_passed": guard_data.get("go_guard") is True,
                      "guard_report_current": guard_current, "cost_table_current": cost_current,
                      "guard_isolated_runs": len(guard_data.get("runs", ())),
                      "cost_cases": len(cost_data.get("cases", ())),
                      "go_stage5_regression": bool(summary and data["go_core"] and guard_current and cost_current)}
        stage5_ok = validation["go_stage5_regression"]
        data["stage5_validation"] = validation
        data_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        with result_path.open("a", encoding="utf-8") as report:
            report.write("\n## 第五阶段验收汇总\n\n")
            report.write(f"- 测试：{validation['pytest_summary']}；退出码 {validation['pytest_exit_code']}。\n")
            report.write(f"- 旧隔离检查：{sum(data['checks'].values())}/{len(data['checks'])}；逐例 {len(data['runs'])}，附加重复 {len(data['repeat_runs'])}。\n")
            report.write(f"- 新 Guard 隔离：{validation['guard_isolated_runs']} 例，报告验收 {'通过' if validation['guard_report_passed'] else '未通过'}；离线成本样例 {validation['cost_cases']}。\n")
            report.write(f"- 当前源码指纹：Guard {'匹配' if guard_current else '失配'}；成本表 {'匹配' if cost_current else '失配'}。\n")
    for item, passed in data["checks"].items():
        print(f"{'通过' if passed else '失败'}：{item}")
    print(f"受限案例验收：{'通过' if data['go_core'] else '未通过'}")
    print(f"A 模板内的动态候选修订：{'已验证' if data['refinement_verified'] else '尚未验证'}")
    print(f"结果文件：{result_path.name}")
    return demo.returncode or (0 if stage5_ok else 1)


if __name__ == "__main__":
    raise SystemExit(main())

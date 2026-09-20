"""一键运行阶段测试和隔离回归，并打印简短验收结果。"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description="一键验证 TritonPact 契约分析与分派")
    parser.add_argument("--quick", action="store_true", help="只运行一轮案例；默认关键案例各运行十次")
    parser.add_argument("--output", type=Path, help="将本次隔离回归另存到指定 Markdown 路径")
    args = parser.parse_args()

    print("[1/2] 运行测试", flush=True)
    tests = subprocess.run([sys.executable, "-m", "pytest", "-q", "test/test_contract_dsl.py", "test/test_access_ir.py", "test/test_stage_integration.py", "test/test_poc.py", "test/test_candidates.py", "test/test_shape_stride.py", "test/test_alignment.py", "test/test_span.py", "test/test_holdouts.py", "test/test_candidate_report.py", "test/test_mutation.py", "test/test_smt_refine.py"], cwd=ROOT, check=False)
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
    for item, passed in data["checks"].items():
        print(f"{'通过' if passed else '失败'}：{item}")
    print(f"受限案例验收：{'通过' if data['go_core'] else '未通过'}")
    print(f"A 模板内的动态候选修订：{'已验证' if data['refinement_verified'] else '尚未验证'}")
    print(f"结果文件：{result_path.name}")
    return demo.returncode


if __name__ == "__main__":
    raise SystemExit(main())

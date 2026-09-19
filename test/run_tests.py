"""一键运行 PoC 测试和隔离案例，并打印简短验收结果。"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description="一键验证 TritonPact PoC")
    parser.add_argument("--quick", action="store_true", help="只运行一轮案例；默认关键案例各运行十次")
    args = parser.parse_args()

    print("[1/2] 运行测试", flush=True)
    tests = subprocess.run([sys.executable, "-m", "pytest", "-q", "test/test_poc.py"], cwd=ROOT, check=False)
    if tests.returncode:
        return tests.returncode

    print("[2/2] 运行隔离案例", flush=True)
    repeats = "1" if args.quick else "10"
    result_path = ROOT / "results" / ("poc_results_quick.md" if args.quick else "poc_results.md")
    demo = subprocess.run([sys.executable, "main.py", "demo", "--repeats", repeats, "--output", str(result_path)], cwd=ROOT, check=False)
    data_path = result_path.with_suffix(".json")
    if not result_path.exists() or not data_path.exists():
        print("没有生成结果文件")
        return demo.returncode or 1

    data = json.loads(data_path.read_text(encoding="utf-8"))
    for item, passed in data["checks"].items():
        print(f"{'通过' if passed else '失败'}：{item}")
    print(f"受限 PoC 最低验收：{'通过' if data['go_core'] else '未通过'}")
    print(f"A 模板内的动态候选修订：{'已验证' if data['refinement_verified'] else '尚未验证'}")
    print(f"结果文件：{result_path.name}")
    return demo.returncode


if __name__ == "__main__":
    raise SystemExit(main())

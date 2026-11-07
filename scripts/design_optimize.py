"""串联可重构臂的多目标搜索。与网页工作台使用同一套模型与 NSGA-II。

在仓库根目录、已安装 NumPy 时运行::

    python scripts/design_optimize.py --mode mixed --pop 16 --gen 8 --samples 300 --seed 1 --out experiments/platform_seed1.json

实物模式不会缩放 ZBot。概念混合模式才会把轴向概念连杆的长度放进搜索。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401

from zbot_exolimb_grasp.platform.exporting import candidates_csv
from zbot_exolimb_grasp.platform.model import baseline_design
from zbot_exolimb_grasp.platform.search import run_search, spec_from_dict, validate_search


def main() -> None:
    parser = argparse.ArgumentParser(description="模块化可重构臂的构型与参数搜索")
    parser.add_argument("--mode", choices=("physical", "mixed"), default="mixed")
    parser.add_argument("--count-min", type=int, default=3)
    parser.add_argument("--count-max", type=int, default=6)
    parser.add_argument("--count-lock", type=int, default=None)
    parser.add_argument("--pop", type=int, default=16)
    parser.add_argument("--gen", type=int, default=8)
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--payload", type=float, default=0.4)
    parser.add_argument("--velocity-limit", type=float, default=1.5)
    parser.add_argument("--effort-limit", type=float, default=20.0)
    parser.add_argument("--objectives", default="coverage,manipulability,mass")
    parser.add_argument("--task-center", nargs=3, type=float, default=(0.0, 0.05, 0.72))
    parser.add_argument("--task-half", nargs=3, type=float, default=(0.22, 0.22, 0.22))
    parser.add_argument("--length-min", type=float, default=0.06)
    parser.add_argument("--length-max", type=float, default=0.16)
    parser.add_argument("--out", type=Path, default=Path("experiments/platform_seed1.json"))
    args = parser.parse_args()
    spec = spec_from_dict(
        {
            "mode": args.mode,
            "count_min": args.count_min,
            "count_max": args.count_max,
            "count_lock": args.count_lock,
            "population": args.pop,
            "generations": args.gen,
            "objectives": tuple(part.strip() for part in args.objectives.split(",") if part.strip()),
            "length_lower_m": args.length_min,
            "length_upper_m": args.length_max,
            "seed": args.seed,
            "task": {
                "center_m": args.task_center,
                "half_m": args.task_half,
                "payload_kg": args.payload,
                "effort_limit_nm": args.effort_limit,
                "effort_is_override": abs(args.effort_limit - 20.0) > 1e-9,
                "velocity_limit_rad_s": args.velocity_limit,
                "sample_count": args.samples,
                "seed": args.seed,
            },
        }
    )
    errors = validate_search(spec)
    if errors:
        raise SystemExit("；".join(errors))
    result = run_search(spec, start=baseline_design(), cloud_front=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    csv_path = args.out.with_suffix(".csv")
    csv_path.write_text(candidates_csv(result), encoding="utf-8")
    baseline = result["baseline"]["metrics"]
    print(
        f"模式 {spec.mode}，评价 {result['evaluations']} 次，用时 {result['elapsed_s']:.1f} s，"
        f"可行前沿 {len(result['feasible_pareto'])} 条"
    )
    print(
        "基准六模块 ZBot："
        f"质量 {baseline['mass_kg']:.3f} kg，覆盖率 {baseline['task_coverage']:.3f} "
        f"± {baseline['coverage_se']:.3f}，可操作度 {baseline['median_translational_manipulability']:.5f}"
    )
    for index, row in enumerate(result["feasible_pareto"], start=1):
        metrics = row["metrics"]
        names = " ".join(item["id"] for item in row["design"]["modules"])
        print(
            f"{index}. {names} | 质量 {metrics['mass_kg']:.3f} | 覆盖率 {metrics['task_coverage']:.3f} "
            f"| 可操作度 {metrics['median_translational_manipulability']:.5f}"
        )
    print(f"已写入 {args.out}")
    print(f"已写入 {csv_path}")


if __name__ == "__main__":
    main()

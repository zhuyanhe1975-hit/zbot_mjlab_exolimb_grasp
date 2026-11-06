"""评估全部合法的离散连接朝向，并写出帕累托摘要。

六个模块之间有五个连接，每个连接只能是 0 或 π/2，合法序列一共 32 条。
这里把这 32 条全部算完，而不是再发明更大的搜索空间。关节采样与
`evaluate_config.py` 相同（种子 0、20000 组、±6.28 rad），因此基准构型和
「中间模块旋转 90°」的数字可以对照。

工作空间用末端原点的体素个数，可操作度是平移雅可比的 Yoshikawa 中位数。
没有速度多面体，也没有 NSGA-II。

用法（在仓库根目录，需要 NumPy）::

    python scripts/evaluate_batch.py

写出：

- web/public/config_batch.csv
- web/public/config_batch_summary.json
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401

import numpy as np

from evaluate_config import (  # noqa: E402
    JOINT_LIMIT,
    SAMPLE_COUNT,
    SEED,
    VOXEL_SIZE_M,
    check_baseline,
    occupied_voxels,
)
from zbot_exolimb_grasp.config_gen import (  # noqa: E402
    INTERFACE_COUNT,
    MIDDLE_INTERFACE,
    forward_kinematics,
    translational_manipulability,
)

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "web" / "public" / "config_batch.csv"
SUMMARY_PATH = ROOT / "web" / "public" / "config_batch_summary.json"

# 与已发布的 config_metrics.json 一致，用来确认采样没有漂。
EXPECTED = {
    "baseline": (3581, 0.00645372),
    "middle": (3580, 0.00606202),
}


def sequences() -> list[tuple[str, tuple[float, ...]]]:
    rows: list[tuple[str, tuple[float, ...]]] = []
    for mask in range(1 << INTERFACE_COUNT):
        orientations = tuple(
            (np.pi / 2) if (mask & (1 << index)) else 0.0 for index in range(INTERFACE_COUNT)
        )
        bits = "".join("1" if (mask & (1 << index)) else "0" for index in range(INTERFACE_COUNT))
        rows.append((bits, orientations))
    rows.sort(key=lambda item: item[0])
    return rows


def identity(bits: str) -> tuple[str, str]:
    if bits == "0" * INTERFACE_COUNT:
        return "baseline", "基准构型"
    middle_bits = ["0"] * INTERFACE_COUNT
    middle_bits[MIDDLE_INTERFACE] = "1"
    if bits == "".join(middle_bits):
        return "middle", "中间模块旋转 90°"
    turned = [str(index + 1) for index, bit in enumerate(bits) if bit == "1"]
    return f"bits-{bits}", "连接 " + "、".join(turned) + " 旋转 90°"


def pareto_flags(rows: list[dict[str, object]]) -> None:
    """同时最大化体素占用和平移可操作度。被其他点支配的序列标成非帕累托。"""
    for index, row in enumerate(rows):
        dominated = False
        voxels = int(row["occupied_voxels"])
        manip = float(row["median_translational_manipulability"])
        for other_index, other in enumerate(rows):
            if other_index == index:
                continue
            other_voxels = int(other["occupied_voxels"])
            other_manip = float(other["median_translational_manipulability"])
            if other_voxels >= voxels and other_manip >= manip and (
                other_voxels > voxels or other_manip > manip
            ):
                dominated = True
                break
        row["pareto"] = not dominated


def select_representatives(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """保留两条对照构型，再从帕累托前沿里最多加两条比基准更好的。"""
    by_id = {str(row["id"]): row for row in rows}
    selected = [dict(by_id["baseline"]), dict(by_id["middle"])]
    for row in selected:
        row["picked_for"] = "对照"
    seen = {str(row["id"]) for row in selected}
    baseline = by_id["baseline"]
    pool = [
        row
        for row in rows
        if row["pareto"]
        and str(row["id"]) not in seen
        and (
            int(row["occupied_voxels"]) > int(baseline["occupied_voxels"])
            or float(row["median_translational_manipulability"])
            > float(baseline["median_translational_manipulability"])
        )
    ]
    if not pool:
        return selected
    by_manip = sorted(
        pool,
        key=lambda row: (
            -float(row["median_translational_manipulability"]),
            -int(row["occupied_voxels"]),
            str(row["bits"]),
        ),
    )
    best_manip = dict(by_manip[0])
    if float(best_manip["median_translational_manipulability"]) > float(
        baseline["median_translational_manipulability"]
    ):
        best_manip["picked_for"] = "可操作度最高"
    else:
        best_manip["picked_for"] = "前沿上可操作度较高"
    selected.append(best_manip)
    seen.add(str(best_manip["id"]))
    by_voxels = sorted(
        pool,
        key=lambda row: (
            -int(row["occupied_voxels"]),
            -float(row["median_translational_manipulability"]),
            str(row["bits"]),
        ),
    )
    if str(by_voxels[0]["id"]) not in seen:
        best_voxels = dict(by_voxels[0])
        best_voxels["picked_for"] = "体素最多"
        selected.append(best_voxels)
    return selected


def evaluate() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    check_baseline()
    rng = np.random.default_rng(SEED)
    samples = rng.uniform(-JOINT_LIMIT, JOINT_LIMIT, size=(SAMPLE_COUNT, 6))
    rows: list[dict[str, object]] = []
    for bits, orientations in sequences():
        config_id, label = identity(bits)
        positions, jacobian = forward_kinematics(samples, orientations)
        manipulability = translational_manipulability(jacobian)
        voxels = occupied_voxels(positions, VOXEL_SIZE_M)
        median = round(float(np.median(manipulability)), 8)
        row = {
            "id": config_id,
            "label": label,
            "bits": bits,
            "orientations_rad": [float(value) for value in orientations],
            "occupied_voxels": voxels,
            "workspace_volume_m3": round(voxels * VOXEL_SIZE_M**3, 6),
            "median_translational_manipulability": median,
        }
        expected = EXPECTED.get(config_id)
        if expected is not None and (voxels, median) != expected:
            raise SystemExit(
                f"{label} 的指标变成了 {(voxels, median)}，与已发布结果 {expected} 不一致"
            )
        rows.append(row)
    pareto_flags(rows)
    return rows, select_representatives(rows)


def write_csv(rows: list[dict[str, object]], representative_ids: set[str]) -> None:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id",
                "label",
                "bits",
                "orientations_rad",
                "occupied_voxels",
                "workspace_volume_m3",
                "median_translational_manipulability",
                "pareto",
                "representative",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "id": row["id"],
                    "label": row["label"],
                    "bits": row["bits"],
                    "orientations_rad": " ".join(f"{value:.8f}" for value in row["orientations_rad"]),
                    "occupied_voxels": row["occupied_voxels"],
                    "workspace_volume_m3": f"{row['workspace_volume_m3']:.6f}",
                    "median_translational_manipulability": f"{row['median_translational_manipulability']:.8f}",
                    "pareto": int(bool(row["pareto"])),
                    "representative": int(str(row["id"]) in representative_ids),
                }
            )


def print_summary(rows: list[dict[str, object]], representatives: list[dict[str, object]]) -> None:
    pareto = [row for row in rows if row["pareto"]]
    print(
        f"合法序列 {len(rows)} 条（五个连接，各为 0 或 π/2）。"
        f"采样 {SAMPLE_COUNT} 组，种子 {SEED}，体素边长 {VOXEL_SIZE_M} m。"
    )
    print(f"帕累托前沿 {len(pareto)} 条（同时看体素占用和可操作度中位数）：")
    print("构型 | 体素占用 | 可操作度中位数 | 连接位")
    for row in pareto:
        print(
            f"{row['label']} | {row['occupied_voxels']} | "
            f"{row['median_translational_manipulability']:.6f} | {row['bits']}"
        )
    print("放进页面的代表构型：")
    for row in representatives:
        print(
            f"- {row['label']}（{row['picked_for']}）体素 {row['occupied_voxels']}，"
            f"可操作度 {row['median_translational_manipulability']:.6f}"
        )


def main() -> None:
    rows, representatives = evaluate()
    representative_ids = {str(row["id"]) for row in representatives}
    write_csv(rows, representative_ids)
    for row in rows:
        row["representative"] = str(row["id"]) in representative_ids
    payload = {
        "sample_count": SAMPLE_COUNT,
        "seed": SEED,
        "voxel_size_m": VOXEL_SIZE_M,
        "joint_range_rad": [-JOINT_LIMIT, JOINT_LIMIT],
        "sequence_count": len(rows),
        "pareto_count": sum(1 for row in rows if row["pareto"]),
        "jacobian": "translational_rows",
        "manipulability": "yoshikawa_sqrt_det_jjt",
        "note": "五个连接各取 0 或 π/2，合法序列共 32 条，已全部评估。基准构型的可操作度中位数最高；另外两条在帕累托前沿上，体素更多、可操作度略低。中间模块旋转 90° 留作对照。",
        "pareto": [row for row in rows if row["pareto"]],
        "representatives": representatives,
        "configs": representatives,
    }
    SUMMARY_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print_summary(rows, representatives)
    print(f"已写入 {CSV_PATH.relative_to(ROOT)}")
    print(f"已写入 {SUMMARY_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

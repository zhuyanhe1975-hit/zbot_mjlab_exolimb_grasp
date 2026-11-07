"""平台核心的回归与搜索测试。不依赖 MJLab。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from zbot_exolimb_grasp.config_gen import BASELINE, forward_kinematics
from zbot_exolimb_grasp.constants import NOMINAL_JOINT_POS, STRAIGHT_JOINT_POS, TOTAL_MASS
from zbot_exolimb_grasp.platform.exporting import assert_mjcf_matches, export_mjcf
from zbot_exolimb_grasp.platform.model import (
    actuation_report,
    assembly_errors,
    baseline_design,
    evaluate_design,
    forward_chain,
    task_targets,
    zbot_chain,
)
from zbot_exolimb_grasp.platform.nsga2 import Individual, dominates, feasible_front, nsga2
from zbot_exolimb_grasp.platform.schema import Design, SearchSpec, TaskSpec
from zbot_exolimb_grasp.platform.search import run_search, validate_search


STRAIGHT_EE = np.array([2.12137298e-4, 0.0, 1.00478799])
NOMINAL_EE = np.array([2.66836957e-4, 0.124544078, 0.455248373])


class BaselineTests(unittest.TestCase):
    def test_straight_and_nominal_match_r3(self) -> None:
        design = baseline_design()
        straight = forward_chain(design, np.array(STRAIGHT_JOINT_POS))["ee"][0]
        nominal = forward_chain(design, np.array(NOMINAL_JOINT_POS))["ee"][0]
        self.assertTrue(np.allclose(straight, STRAIGHT_EE, atol=1e-8))
        self.assertTrue(np.allclose(nominal, NOMINAL_EE, atol=1e-8))
        reference, _ = forward_kinematics(np.array(NOMINAL_JOINT_POS), BASELINE)
        self.assertTrue(np.allclose(nominal, reference, atol=1e-8))

    def test_mass_matches_six_module_arm(self) -> None:
        self.assertAlmostEqual(forward_chain(baseline_design(), np.zeros((1, 6)))["mass_kg"], TOTAL_MASS, places=5)

    def test_jacobian_finite_difference(self) -> None:
        design = zbot_chain(4, (0.0, math_pi() / 2, 0.0))
        rng = np.random.default_rng(2)
        pose = rng.uniform(-1.5, 1.5, size=4)
        base = forward_chain(design, pose)
        jac = base["jacobian"][0]
        numeric = np.zeros((3, 4))
        step = 1e-6
        for index in range(4):
            bumped = pose.copy()
            bumped[index] += step
            numeric[:, index] = (forward_chain(design, bumped)["ee"][0] - base["ee"][0]) / step
        self.assertLess(np.max(np.abs(jac - numeric)), 1e-5)

    def test_length_changes_tip_and_mass(self) -> None:
        def chain(length: float) -> Design:
            return Design(
                ("zbot_r3", "concept_axial", "zbot_r3"),
                (0.0, 0.0),
                ({}, {"length_m": length}, {}),
            )

        short = chain(0.08)
        long = chain(0.16)
        self.assertGreater(forward_chain(long, np.zeros(3))["mass_kg"], forward_chain(short, np.zeros(3))["mass_kg"])
        pose = np.array([0.4, 0.7, -0.2])
        short_ee = forward_chain(short, pose)["ee"][0]
        long_ee = forward_chain(long, pose)["ee"][0]
        self.assertGreater(np.linalg.norm(long_ee - short_ee), 0.05)


class RuleTests(unittest.TestCase):
    def test_cross_family_is_illegal(self) -> None:
        design = Design(
            ("zbot_r3", "demo_unreleased"),
            (0.0,),
            ({}, {"length_m": 0.1}),
        )
        errors = assembly_errors(design)
        self.assertTrue(any("接口族" in item for item in errors))

    def test_illegal_twist_and_count(self) -> None:
        design = Design(("zbot_r3", "zbot_r3"), (0.3,), ({}, {}))
        self.assertTrue(any("方位" in item for item in errors) for errors in [assembly_errors(design)])
        short = Design(("zbot_r3", "zbot_r3"), (0.0,), ({}, {}))
        self.assertTrue(any("数量" in item for item in assembly_errors(short)))

    def test_physical_rejects_size_parameter(self) -> None:
        design = Design(("zbot_r3", "zbot_r3", "zbot_r3"), (0.0, 0.0), ({"length_m": 1.0}, {}, {}))
        self.assertTrue(any("实物" in item for item in assembly_errors(design)))

    def test_search_input_validation(self) -> None:
        spec = SearchSpec(population=3, generations=0, task=TaskSpec(sample_count=4, payload_kg=-1))
        errors = validate_search(spec)
        self.assertGreaterEqual(len(errors), 3)


class ConstraintTests(unittest.TestCase):
    def test_baseline_q0_is_feasible_under_default_payload(self) -> None:
        metrics = evaluate_design(baseline_design(), TaskSpec(sample_count=64, seed=0))
        self.assertFalse(metrics["q0_collision"])
        self.assertTrue(metrics["q0_torque_ok"])
        self.assertEqual(metrics["unsupported"][0], "网格精确碰撞")

    def test_infeasible_not_in_front(self) -> None:
        feasible = Individual(genome="a", objectives=[0.2, 0.2], violation=0.0)
        infeasible = Individual(genome="b", objectives=[0.0, 0.0], violation=0.4)
        self.assertTrue(dominates(feasible, infeasible))
        self.assertFalse(dominates(infeasible, feasible))


class SearchTests(unittest.TestCase):
    def test_optimizer_is_deterministic_and_respects_constraints(self) -> None:
        spec = SearchSpec(
            mode="mixed",
            count_min=3,
            count_max=4,
            population=6,
            generations=2,
            seed=7,
            task=TaskSpec(sample_count=0, seed=7, payload_kg=0.2, grid_divisions=2, ik_starts=2, ik_iterations=4),
        )
        first = run_search(spec, start=baseline_design())
        second = run_search(spec, start=baseline_design())
        self.assertEqual(first["feasible_pareto"], second["feasible_pareto"])
        for row in first["feasible_pareto"]:
            self.assertEqual(row["violation"], 0.0)
            self.assertEqual(row["metrics"]["feasible"], True)
        self.assertIn("mass_kg", first["baseline"]["metrics"])

    def test_parameter_change_changes_evaluation(self) -> None:
        left = Design(
            ("zbot_r3", "concept_axial", "zbot_r3"),
            (0.0, 0.0),
            ({}, {"length_m": 0.08}, {}),
        )
        right = Design(
            ("zbot_r3", "concept_axial", "zbot_r3"),
            (0.0, 0.0),
            ({}, {"length_m": 0.15}, {}),
        )
        task = TaskSpec(sample_count=64, seed=3)
        a = evaluate_design(left, task)
        b = evaluate_design(right, task)
        self.assertGreater(b["mass_kg"], a["mass_kg"] + 0.2)
        self.assertGreater(np.linalg.norm(np.array(a["ee_straight"]) - np.array(b["ee_straight"])), 0.05)

    def test_tiny_problem_improves_or_stays_nondominated(self) -> None:
        def evaluate(genomes: list[tuple[float, float]]):
            rows = []
            for x, y in genomes:
                rows.append(([x, y], 0.0 if x >= 0 else 1.0, {}))
            return rows

        def initialize(rng, population):
            return [tuple(rng.uniform(-0.2, 1.0, size=2)) for _ in range(population)]

        def vary(left, right, rng):
            child_a = tuple(float(np.clip(0.5 * (a + b) + rng.normal(0, 0.05), -0.2, 1)) for a, b in zip(left, right))
            child_b = tuple(float(np.clip(0.5 * (a + b) + rng.normal(0, 0.05), -0.2, 1)) for a, b in zip(right, left))
            return child_a, child_b

        pop = nsga2(population=8, generations=6, seed=1, initialize=initialize, vary=vary, evaluate=evaluate)
        front = feasible_front(pop)
        self.assertTrue(front)
        self.assertTrue(all(item.violation == 0.0 and item.objectives[0] >= -1e-9 for item in front))


class SixDofTests(unittest.TestCase):
    def test_module_count_is_driven_joint_count(self) -> None:
        baseline = actuation_report(baseline_design())
        self.assertEqual(baseline["driven_joint_count"], 6)
        self.assertEqual(baseline["module_count"], 6)
        self.assertEqual(baseline["passive_body_count"], 6)
        mixed = Design(
            ("zbot_r3", "zbot_r3", "zbot_r3", "zbot_r3", "zbot_r3", "concept_axial"),
            (0.0, 0.0, 0.0, 0.0, 0.0),
            ({}, {}, {}, {}, {}, {"length_m": 0.1}),
        )
        report = actuation_report(mixed)
        self.assertEqual(report["driven_joint_count"], 6)
        self.assertEqual(report["passive_body_count"], 5)
        self.assertEqual(actuation_report(zbot_chain(4))["driven_joint_count"], 4)

    def test_physical_six_has_exactly_32_legal_twists(self) -> None:
        seen = set()
        for mask in range(32):
            twists = tuple((math_pi() / 2 if mask & (1 << index) else 0.0) for index in range(5))
            design = zbot_chain(6, twists)
            self.assertEqual(assembly_errors(design), [])
            self.assertEqual(actuation_report(design)["driven_joint_count"], 6)
            seen.add(tuple(round(value, 6) for value in design.twists_rad))
        self.assertEqual(len(seen), 32)
        self.assertIn((0.0, 0.0, 0.0, 0.0, 0.0), seen)


class MetricTests(unittest.TestCase):
    def test_same_task_grid_for_different_module_counts(self) -> None:
        task = TaskSpec(grid_divisions=3, sample_count=0, ik_starts=1, ik_iterations=1)
        targets = task_targets(task)
        self.assertEqual(targets.shape, (27, 3))
        again = task_targets(task)
        self.assertTrue(np.allclose(targets, again))
        short = evaluate_design(zbot_chain(3), task)
        long = evaluate_design(zbot_chain(6), task)
        self.assertEqual(short["task_target_count"], long["task_target_count"])
        self.assertEqual(short["position_tolerance_m"], long["position_tolerance_m"])
        self.assertNotIn("task_coverage", short)

    def test_clustered_reach_is_not_full_spatial_coverage(self) -> None:
        design = Design(
            ("concept_axial", "concept_axial", "concept_axial"),
            (0.0, 0.0),
            ({"length_m": 0.06}, {"length_m": 0.06}, {"length_m": 0.06}),
        )
        task = TaskSpec(
            center_m=(0.2, 0.2, 0.7),
            half_m=(0.4, 0.4, 0.4),
            grid_divisions=3,
            ik_starts=3,
            ik_iterations=8,
            position_tolerance_m=0.05,
            sample_count=80,
            seed=1,
            payload_kg=0.0,
        )
        metrics = evaluate_design(design, task)
        self.assertLess(metrics["task_feasible_fraction"], 0.5)
        self.assertIsNotNone(metrics["pose_hit_rate"])
        self.assertGreater(metrics["pose_hit_rate"], metrics["task_feasible_fraction"])

    def test_infeasible_poses_do_not_count_as_task_coverage(self) -> None:
        task = TaskSpec(
            sample_count=60,
            seed=0,
            effort_limit_nm=0.05,
            payload_kg=2.0,
            grid_divisions=2,
            ik_starts=2,
            ik_iterations=6,
        )
        metrics = evaluate_design(baseline_design(), task)
        self.assertGreater(metrics["pose_hit_rate"], 0.0)
        self.assertEqual(metrics["task_feasible_fraction"], 0.0)
        self.assertFalse(metrics["task_manipulability_valid"])
        self.assertIsNone(metrics["task_manipulability_median"])
        self.assertTrue(metrics["feasible"])
        self.assertFalse(metrics["q0_is_design_veto"])

    def test_resolution_changes_denominator_without_claiming_monotonicity(self) -> None:
        design = zbot_chain(4)
        coarse_task = TaskSpec(grid_divisions=2, ik_starts=2, ik_iterations=5, sample_count=0)
        fine_task = TaskSpec(grid_divisions=3, ik_starts=2, ik_iterations=5, sample_count=0)
        coarse = evaluate_design(design, coarse_task)
        fine = evaluate_design(design, fine_task)
        self.assertEqual(coarse["task_target_count"], 8)
        self.assertEqual(fine["task_target_count"], 27)
        self.assertGreaterEqual(coarse["task_feasible_fraction"], 0.0)
        self.assertLessEqual(fine["task_feasible_fraction"], 1.0)


class ExportTests(unittest.TestCase):
    def test_mjcf_matches_internal_fk(self) -> None:
        assert_mjcf_matches(baseline_design(), samples=3, seed=1)
        mixed = Design(
            ("zbot_r3", "concept_axial", "zbot_r3"),
            (math_pi() / 2, 0.0),
            ({}, {"length_m": 0.11}, {}),
        )
        assert_mjcf_matches(mixed, samples=3, seed=2)
        xml = export_mjcf(baseline_design())
        self.assertEqual(xml.count("<joint "), 6)
        self.assertIn('axis="0 0 1"', xml)


def math_pi() -> float:
    return float(np.pi)


if __name__ == "__main__":
    unittest.main()

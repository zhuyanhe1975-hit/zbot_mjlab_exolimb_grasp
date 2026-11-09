"""科研后端回归。缺可选依赖时跳过，不改自写模型的测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "generative-robot-design"))

from zbot_exolimb_grasp.platform.research_backend import availability


def _ready() -> bool:
    status = availability()
    return all(isinstance(status.get(name), str) and not str(status[name]).startswith("unavailable") for name in ("torch", "mujoco"))


@unittest.skipUnless(_ready(), "科研后端的 torch / mujoco 没装，跳过")
class ResearchBackendTests(unittest.TestCase):
    def test_upstream_transform_matches_the_factored_product(self) -> None:
        from zbot_exolimb_grasp.platform.research_backend import module_matrices, scalar_module_matches_product

        proximal, distal, _mass = module_matrices(("zbot_r3",) * 6, (0.0,) * 5, (0.0,) * 6)
        self.assertLess(scalar_module_matches_product(proximal, distal, 0.4), 1e-12)

    def test_six_module_matches_mujoco_xml(self) -> None:
        from zbot_exolimb_grasp.constants import NOMINAL_JOINT_POS
        from zbot_exolimb_grasp.platform.research_backend import module_matrices, mujoco_tool_positions, positions

        proximal, distal, _mass = module_matrices(("zbot_r3",) * 6, (0.0,) * 5, (0.0,) * 6)
        xml = ROOT / "assets" / "zbot_6" / "ZBOT6.xml"
        poses = [np.zeros(6), np.asarray(NOMINAL_JOINT_POS), np.linspace(-1.0, 0.8, 6), np.array([0.4, -0.7, 1.1, -0.4, 0.6, -0.2])]
        for pose in poses:
            guess_origin = positions(proximal, distal, pose, "foot_joint_origin")
            guess_sole = positions(proximal, distal, pose, "foot_1_sole")
            reference = mujoco_tool_positions(xml, pose, site_name="foot_1_sole", body_name="foot_1")
            self.assertLess(float(np.max(np.abs(guess_origin - reference["body"]))), 1e-6)
            self.assertLess(float(np.max(np.abs(guess_sole - reference["site"]))), 1e-6)

    def test_seven_and_eight_match_exported_mjcf(self) -> None:
        from zbot_exolimb_grasp.platform.exporting import export_mjcf
        from zbot_exolimb_grasp.platform.model import zbot_chain
        from zbot_exolimb_grasp.platform.research_backend import module_matrices, mujoco_tool_positions, positions

        for count in (7, 8):
            design = zbot_chain(count)
            proximal, distal, _mass = module_matrices(("zbot_r3",) * count, (0.0,) * (count - 1), (0.0,) * count)
            xml = export_mjcf(design, tool_point="foot_1_sole")
            pose = np.linspace(-0.6, 0.5, count)
            guess = positions(proximal, distal, pose, "foot_1_sole")
            reference = mujoco_tool_positions(xml, pose, site_name="ee")
            self.assertLess(float(np.max(np.abs(guess - reference["site"]))), 1e-6)


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from pathlib import Path

import numpy as np

from export.bvh_exporter import BVHExporter
from retargeting.mixamo_retargeter import MixamoRetargeter


class TestRetargetAndExport(unittest.TestCase):
    def test_retarget_shapes(self) -> None:
        pose = np.random.randn(20, 17, 3).astype(np.float32)
        retargeter = MixamoRetargeter()
        anim = retargeter.retarget(pose)
        self.assertEqual(anim.local_quaternions.shape, (20, 19, 4))
        self.assertEqual(anim.root_positions.shape, (20, 3))
        self.assertIn("LeftHand", anim.joint_names)
        self.assertIn("RightHand", anim.joint_names)

    def test_bvh_export(self) -> None:
        pose = np.random.randn(5, 17, 3).astype(np.float32)
        anim = MixamoRetargeter().retarget(pose)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "test.bvh"
            BVHExporter(frame_time=1 / 30).export(anim, p)
            text = p.read_text(encoding="utf-8")
            self.assertIn("HIERARCHY", text)
            self.assertIn("MOTION", text)


if __name__ == "__main__":
    unittest.main()

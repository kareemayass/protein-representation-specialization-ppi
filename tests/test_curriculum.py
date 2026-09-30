from collections import Counter
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from interpro_joint.curriculum import JointCurriculum


class CurriculumTests(unittest.TestCase):
    def test_task_ratios_and_phase_boundaries(self):
        curriculum = JointCurriculum(400, seed=47)
        tasks = [curriculum.task(i) for i in range(400)]
        self.assertEqual(set(tasks[:60]), {"stage1"})
        for start in range(60, 140, 4):
            self.assertEqual(Counter(tasks[start:start + 4]), {"stage1": 3, "stage2": 1})
        for start in range(140, 400, 2):
            self.assertEqual(Counter(tasks[start:start + 2]), {"stage1": 1, "stage2": 1})
        self.assertEqual(Counter(tasks), {"stage1": 250, "stage2": 150})

    def test_rank_and_resume_consistency(self):
        ranks = [JointCurriculum(400, seed=47) for _ in range(4)]
        for step in [0, 59, 60, 139, 140, 237, 399]:
            self.assertEqual(len({rank.task(step) for rank in ranks}), 1)
        resumed = JointCurriculum(400, seed=47)
        self.assertEqual([resumed.task(i) for i in range(237, 400)], [ranks[0].task(i) for i in range(237, 400)])

    def test_invalid_steps(self):
        for total in [0, -40, 39, 41]:
            with self.subTest(total=total), self.assertRaises(ValueError):
                JointCurriculum(total)
        curriculum = JointCurriculum(40)
        for step in [-1, 40]:
            with self.subTest(step=step), self.assertRaises(IndexError):
                curriculum.task(step)


if __name__ == "__main__":
    unittest.main()

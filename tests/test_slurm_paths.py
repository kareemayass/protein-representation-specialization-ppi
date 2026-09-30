"""Regression check: artifact directories must not hide repository modules."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which("bash"), "Bash is required for Slurm path checks")
class SlurmPathTests(unittest.TestCase):
    def test_imports_with_separate_artifact_directory(self):
        with tempfile.TemporaryDirectory(prefix="ppi artifacts ") as external:
            for path in sorted((ROOT / "scripts").rglob("*.slurm")):
                with self.subTest(script=path.name):
                    # Evaluate only path exports, not module loads, srun, or jobs.
                    exports = [line for line in path.read_text().splitlines() if line.startswith("export PYTHONPATH=")]
                    self.assertTrue(exports)
                    command = "\n".join(exports) + '\n"$PYTHON_BIN" -c "from interpro_joint.curriculum import JointCurriculum; assert JointCurriculum(40).summarize().stage1_steps == 25"'
                    env = {**os.environ, "REPO_ROOT": str(ROOT), "PROJECT": external,
                           "PYTHONPATH": "", "PYTHON_BIN": sys.executable}
                    result = subprocess.run(["bash", "-eu", "-c", command], cwd=external, env=env, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()

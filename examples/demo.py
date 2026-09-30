"""Run a synthetic prediction audit and inspect the actual training curriculum.

No model weights, downloads, third-party packages, or GPU are needed.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from interpro_joint.curriculum import JointCurriculum
from ppi_audit.predictions import compare_predictions, read_predictions


def main() -> None:
    audit = compare_predictions(
        read_predictions(ROOT / "examples/toy_baseline.tsv"),
        read_predictions(ROOT / "examples/toy_adapted.tsv"),
    )
    print(json.dumps({
        "data_status": "SYNTHETIC demonstration; not experimental results or PPI inference",
        "prediction_audit": audit,
        "research_curriculum_40_step_example": JointCurriculum(40, seed=47).summarize().as_dict(),
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

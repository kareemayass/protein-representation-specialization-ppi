import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ppi_audit.predictions import Prediction, binary_metrics, compare_predictions, read_predictions, split_overlap


class PredictionTests(unittest.TestCase):
    def read_tsv(self, body):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.tsv"
            path.write_text("protein_A\tprotein_B\tlabel\tscore\n" + body, encoding="utf-8")
            return read_predictions(path)

    def test_shuffled_reversed_pairs_align_and_changes_balance(self):
        result = compare_predictions(
            read_predictions(ROOT / "examples/toy_baseline.tsv"),
            read_predictions(ROOT / "examples/toy_adapted.tsv"),
        )
        self.assertEqual(result["baseline"]["accuracy"], 5 / 6)
        self.assertEqual(result["adapted"]["accuracy"], 5 / 6)
        self.assertEqual(result["transitions"], dict(corrected=1, broken=1, unchanged_correct=4, unchanged_incorrect=0))
        self.assertEqual(result["net_additional_correct"], 0)
        self.assertGreater(result["adapted"]["recall"], result["baseline"]["recall"])
        self.assertLess(result["adapted"]["specificity"], result["baseline"]["specificity"])

    def test_duplicate_reversed_pair_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate unordered"):
            self.read_tsv("A\tB\t1\t0.8\nB\tA\t1\t0.8\n")

    def test_ids_preserved_without_delimiter_collisions(self):
        pairs = self.read_tsv("001\tA||B\t1\t0.8\n001||A\tB\t0\t0.1\n")
        self.assertEqual(len(pairs), 2)
        self.assertIn(("001", "A||B"), pairs)

    def test_invalid_scores_and_labels_rejected(self):
        for value in ["nan", "inf", "-inf", "1.1", "-0.1", "", "oops"]:
            with self.subTest(score=value), self.assertRaises(ValueError):
                self.read_tsv(f"A\tB\t1\t{value}\n")
        for value in ["0.5", "1.0", "2", "", "-1"]:
            with self.subTest(label=value), self.assertRaises(ValueError):
                self.read_tsv(f"A\tB\t{value}\t0.5\n")

    def test_empty_malformed_and_missing_ids_rejected(self):
        for body in ["", "A\tB\t1\n", "\tB\t1\t0.5\n", "A\tB\t1\t0.5\textra\n"]:
            with self.subTest(body=body), self.assertRaises(ValueError):
                self.read_tsv(body)

    def test_pair_mismatch_and_label_conflict_rejected(self):
        baseline = {("A", "B"): Prediction(1, 0.8)}
        with self.assertRaisesRegex(ValueError, "Pair sets differ"):
            compare_predictions(baseline, {("A", "C"): Prediction(1, 0.8)})
        with self.assertRaisesRegex(ValueError, "Labels disagree"):
            compare_predictions(baseline, {("A", "B"): Prediction(0, 0.8)})

    def test_threshold_boundary_and_undefined_metrics(self):
        result = binary_metrics({("A", "B"): Prediction(1, 0.5)}, 0.5)
        self.assertEqual(result["tp"], 1)
        self.assertIsNone(result["specificity"])
        self.assertIsNone(result["mcc"])
        for threshold in [math.nan, math.inf, -0.1, 1.1]:
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                binary_metrics({("A", "B"): Prediction(1, 0.5)}, threshold)

    def test_perfect_and_inverted_mcc(self):
        self.assertEqual(binary_metrics(self.read_tsv("A\tB\t1\t1\nC\tD\t0\t0\n"), 0.5)["mcc"], 1)
        self.assertEqual(binary_metrics(self.read_tsv("A\tB\t1\t0\nC\tD\t0\t1\n"), 0.5)["mcc"], -1)

    def test_protein_overlap_without_pair_overlap(self):
        result = split_overlap({
            "train": {("A", "B"): Prediction(1, 0.8)},
            "test": {("A", "C"): Prediction(0, 0.2)},
        })
        self.assertFalse(result["protein_disjoint"])
        self.assertEqual(result["comparisons"]["train vs test"], {"shared_pairs": 0, "shared_proteins": 1})

    def test_disjoint_split(self):
        self.assertTrue(split_overlap({
            "train": {("A", "B"): Prediction(1, 0.8)},
            "test": {("C", "D"): Prediction(0, 0.2)},
        })["protein_disjoint"])

    def test_cli_json_and_overlap_exit_status(self):
        env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
        args = [str(ROOT / "examples/toy_baseline.tsv"), str(ROOT / "examples/toy_adapted.tsv")]
        compared = subprocess.run([sys.executable, "-m", "ppi_audit", "compare", *args], env=env, capture_output=True, text=True)
        self.assertEqual(compared.returncode, 0, compared.stderr)
        self.assertEqual(json.loads(compared.stdout)["net_additional_correct"], 0)
        overlap = subprocess.run([sys.executable, "-m", "ppi_audit", "splits", *args], env=env, capture_output=True, text=True)
        self.assertEqual(overlap.returncode, 1, overlap.stderr)
        self.assertFalse(json.loads(overlap.stdout)["protein_disjoint"])


if __name__ == "__main__":
    unittest.main()

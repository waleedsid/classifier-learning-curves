import importlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import experiment


class ExperimentTests(unittest.TestCase):
    def test_plateau_uses_start_before_two_small_gains(self):
        self.assertEqual(experiment.diminishing_returns([10, 20, 30, 40], [.7, .8, .803, .805]), 20)

    def test_no_plateau_with_one_small_gain(self):
        self.assertIsNone(experiment.diminishing_returns([10, 20, 30], [.7, .8, .801]))

    def test_declines_count_under_original_signed_rule(self):
        self.assertEqual(experiment.diminishing_returns([10, 20, 30], [.8, .79, .78]), 10)

    def test_misaligned_vectors_rejected(self):
        with self.assertRaises(ValueError):
            experiment.diminishing_returns([1], [.1, .2])

    def test_import_has_no_output_side_effect(self):
        with patch.object(Path, "mkdir", side_effect=AssertionError("Import wrote a folder")):
            importlib.reload(experiment)

    def test_full_run_artifacts_and_original_results(self):
        # Full integration check consumes the artifacts from the documented run.
        root = Path(__file__).resolve().parents[1]
        result = pd.read_csv(root/"results/learning_curve_results.csv")
        original = pd.read_csv(root/"docs/course_results.csv")
        self.assertEqual(len(result), 60)
        self.assertFalse(result.isna().any().any())
        np.testing.assert_allclose(result["Generalization Gap"], result["Train Mean"]-result["Validation Mean"])
        for column in ["Train Mean", "Validation Mean", "Validation SD"]:
            np.testing.assert_allclose(result[column], original[column], atol=1e-8, rtol=0)
        self.assertEqual(result["Training Size"].min(), 143)
        self.assertEqual(result["Training Size"].max(), 1437)
        for image in ["accuracy_learning_curves.png", "macro_f1_learning_curves.png", "generalization_gap.png"]:
            self.assertGreater((root/"results"/image).stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()

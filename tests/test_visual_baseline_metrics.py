from __future__ import annotations

import unittest

from tools import evaluate_visual_model_baselines as model_metrics
from tools import evaluate_visual_shortcut_baselines as shortcut_metrics


class VisualBaselineMetricTests(unittest.TestCase):
    def test_tied_auc_uses_midranks(self) -> None:
        for module in (model_metrics, shortcut_metrics):
            self.assertEqual(module._roc_auc([0.5], [0.5]), 0.5)

    def test_tar_at_far_maximizes_tar_within_constraint(self) -> None:
        positive = [0.95, 0.85, 0.75]
        negative = [0.9, 0.2, 0.1]
        for module in (model_metrics, shortcut_metrics):
            self.assertAlmostEqual(module._tar_at_far(positive, negative, 0.0), 1 / 3)

    def test_eer_uses_discrete_crossing(self) -> None:
        positive = [0.9, 0.4]
        negative = [0.6, 0.2]
        for module in (model_metrics, shortcut_metrics):
            self.assertEqual(module._eer(positive, negative), 0.5)


if __name__ == "__main__":
    unittest.main()

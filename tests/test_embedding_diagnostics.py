from __future__ import annotations

import json
import unittest

import numpy as np

from cvi.evaluation import (
    EMBEDDING_DIAGNOSTICS_SCHEMA_VERSION,
    EmbeddingDiagnosticsConfig,
    EmbeddingDiagnosticsError,
    compute_embedding_diagnostics,
)


class EmbeddingDiagnosticsValidationTest(unittest.TestCase):
    def test_rejects_invalid_embedding_matrices(self):
        invalid = (
            np.empty((0, 3), dtype=np.float64),
            np.empty((3, 0), dtype=np.float64),
            np.ones(3, dtype=np.float64),
            np.array([[1.0, np.nan]]),
            np.array([[1.0, np.inf]]),
            np.array([["not", "numeric"]]),
        )
        for embeddings in invalid:
            with self.subTest(shape=embeddings.shape, dtype=embeddings.dtype):
                with self.assertRaises(EmbeddingDiagnosticsError):
                    compute_embedding_diagnostics(embeddings)

    def test_rejects_zero_and_near_zero_rows(self):
        config = EmbeddingDiagnosticsConfig(minimum_row_norm=1e-6)
        for row in (np.array([0.0, 0.0]), np.array([1e-7, 0.0])):
            with self.subTest(row=row):
                with self.assertRaisesRegex(
                    EmbeddingDiagnosticsError, "minimum_row_norm"
                ):
                    compute_embedding_diagnostics(row[None, :], config=config)

    def test_rejects_invalid_metadata_ids(self):
        embeddings = np.eye(3, dtype=np.float64)
        unhashable = np.empty(3, dtype=object)
        unhashable[:] = ["a", [], "c"]
        invalid_ids = (
            np.array([["a", "b", "c"]]),
            np.array(["a", "b"]),
            np.array(["a", None, "c"], dtype=object),
            np.array(["a", np.nan, "c"], dtype=object),
            np.array(["a", "", "c"]),
            unhashable,
        )
        for ids in invalid_ids:
            with self.subTest(ids=repr(ids)):
                with self.assertRaises(EmbeddingDiagnosticsError):
                    compute_embedding_diagnostics(embeddings, identity_ids=ids)

    def test_rejects_invalid_quality_scores(self):
        embeddings = np.eye(3, dtype=np.float64)
        for scores in (
            np.array([1.0, 2.0]),
            np.array([[1.0, 2.0, 3.0]]),
            np.array([1.0, np.nan, 3.0]),
            np.array(["low", "medium", "high"]),
        ):
            with self.subTest(scores=repr(scores)):
                with self.assertRaises(EmbeddingDiagnosticsError):
                    compute_embedding_diagnostics(
                        embeddings, quality_scores=scores
                    )


class EmbeddingDiagnosticsMetricsTest(unittest.TestCase):
    def test_directional_metrics_are_row_scale_invariant(self):
        embeddings = np.array(
            [
                [1.0, 0.2, 0.0],
                [0.8, 0.4, 0.1],
                [0.0, 1.0, 0.2],
                [0.1, 0.8, 0.5],
                [0.3, 0.1, 1.0],
                [0.4, 0.2, 0.9],
            ]
        )
        scales = np.array([0.5, 2.0, 3.0, 1.5, 4.0, 0.75])
        base = compute_embedding_diagnostics(embeddings)
        scaled = compute_embedding_diagnostics(embeddings * scales[:, None])

        base_geometry = base["normalized_directional_geometry"]
        scaled_geometry = scaled["normalized_directional_geometry"]
        self.assertAlmostEqual(
            base_geometry["centroid_norm"], scaled_geometry["centroid_norm"]
        )
        base_cosines = base_geometry["off_diagonal_cosine"]["summary"]
        scaled_cosines = scaled_geometry["off_diagonal_cosine"]["summary"]
        self.assertEqual(base_cosines["count"], scaled_cosines["count"])
        for name in set(base_cosines) - {"count"}:
            self.assertAlmostEqual(base_cosines[name], scaled_cosines[name])
        self.assertEqual(base["hubness"], scaled["hubness"])
        base_spectrum = base["normalized_centered_covariance_spectrum"]
        scaled_spectrum = scaled["normalized_centered_covariance_spectrum"]
        for name in (
            "effective_rank_entropy",
            "participation_ratio",
            "top_eigenvalue_fraction",
        ):
            self.assertAlmostEqual(base_spectrum[name], scaled_spectrum[name])
        self.assertNotEqual(
            base["raw_norm_summary"]["mean"],
            scaled["raw_norm_summary"]["mean"],
        )

    def test_repeat_dispersion_separates_low_and_high_noise(self):
        repeat_ids = np.array(["sample-a", "sample-a", "sample-b", "sample-b"])
        low_noise = np.array(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
        )
        high_noise = np.array(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [-1.0, 0.0]]
        )

        low = compute_embedding_diagnostics(low_noise, repeat_ids=repeat_ids)
        high = compute_embedding_diagnostics(high_noise, repeat_ids=repeat_ids)

        low_mean = low["repeat_noise"][
            "within_repeat_directional_dispersion"
        ]["mean"]
        high_mean = high["repeat_noise"][
            "within_repeat_directional_dispersion"
        ]["mean"]
        self.assertEqual(low_mean, 0.0)
        self.assertGreater(high_mean, low_mean)

        magnitude_noise = compute_embedding_diagnostics(
            np.array([[1.0, 0.0], [4.0, 0.0], [0.0, 2.0], [0.0, 2.0]]),
            repeat_ids=repeat_ids,
        )["repeat_noise"]["within_repeat_log_norm_standard_deviation"]
        self.assertGreater(magnitude_noise["maximum"], 0.0)

    def test_session_and_domain_metrics_report_coverage(self):
        embeddings = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.8, 0.2, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.9, 0.1],
                [0.0, 0.0, 1.0],
                [0.1, 0.0, 0.9],
            ]
        )
        identity_ids = np.array(["a", "a", "b", "b", "c", "c"])
        session_ids = np.array(["s1", "s2", "s1", "s1", "s3", "s3"])
        domain_ids = np.array(["day", "night", "day", "day", "night", "night"])

        report = compute_embedding_diagnostics(
            embeddings,
            identity_ids=identity_ids,
            session_ids=session_ids,
            domain_ids=domain_ids,
        )

        session = report["session_conditioned"]
        self.assertTrue(session["available"])
        self.assertEqual(session["cross_session_identity_count"], 1)
        self.assertEqual(session["covered_sample_count"], 2)
        domain = report["domain_conditioned"]
        self.assertTrue(domain["available"])
        self.assertIn("confounded", domain["confounding_warning"])
        self.assertEqual(
            domain["identity_overlap_coverage"]["cross_domain_identity_count"],
            1,
        )
        matched_domain = domain["same_identity_cross_domain_shift"]
        self.assertTrue(matched_domain["available"])
        self.assertEqual(matched_domain["cross_domain_identity_count"], 1)
        self.assertIn("session", matched_domain["confounding_warning"])

        absent = compute_embedding_diagnostics(
            embeddings, identity_ids=identity_ids
        )
        self.assertFalse(absent["session_conditioned"]["available"])
        self.assertFalse(absent["domain_conditioned"]["available"])
        one_domain = compute_embedding_diagnostics(
            embeddings, domain_ids=np.repeat("day", len(embeddings))
        )["domain_conditioned"]
        self.assertFalse(one_domain["available"])
        self.assertEqual(one_domain["domain_count"], 1)
        self.assertIn("confounded", one_domain["confounding_warning"])

    def test_identity_metrics_require_repeated_identities(self):
        embeddings = np.array(
            [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]]
        )
        report = compute_embedding_diagnostics(
            embeddings, identity_ids=np.array(["a", "a", "b", "b"])
        )["identity_conditioned"]
        self.assertTrue(report["available"])
        self.assertEqual(report["repeated_identity_count"], 2)
        self.assertTrue(report["between_identity_centroid_cosine"]["available"])

        unavailable = compute_embedding_diagnostics(
            embeddings, identity_ids=np.array(["a", "b", "c", "d"])
        )["identity_conditioned"]
        self.assertFalse(unavailable["available"])
        self.assertIn("multiple samples", unavailable["reason"])

    def test_magnitude_quality_spearman_uses_raw_norms(self):
        embeddings = np.array(
            [[1.0, 0.0], [0.0, 2.0], [3.0, 0.0], [0.0, 4.0]]
        )
        report = compute_embedding_diagnostics(
            embeddings, quality_scores=np.array([10.0, 20.0, 30.0, 40.0])
        )
        association = report["magnitude_quality_association"]
        self.assertTrue(association["available"])
        self.assertAlmostEqual(association["spearman_rank_correlation"], 1.0)

    def test_effective_rank_distinguishes_line_and_isotropic_data(self):
        rank_one = np.array(
            [[1.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0],
             [-1.0, 0.0, 0.0, 0.0], [-2.0, 0.0, 0.0, 0.0]]
        )
        isotropic = np.concatenate((np.eye(4), -np.eye(4)), axis=0)

        low = compute_embedding_diagnostics(rank_one)[
            "centered_covariance_spectrum"
        ]
        high = compute_embedding_diagnostics(isotropic)[
            "centered_covariance_spectrum"
        ]

        self.assertEqual(low["numerical_rank"], 1)
        self.assertAlmostEqual(low["effective_rank_entropy"], 1.0)
        self.assertAlmostEqual(low["participation_ratio"], 1.0)
        self.assertAlmostEqual(high["effective_rank_entropy"], 4.0)
        self.assertAlmostEqual(high["participation_ratio"], 4.0)
        self.assertLess(high["top_eigenvalue_fraction"], low["top_eigenvalue_fraction"])

    def test_report_is_deterministic_bounded_and_json_serializable(self):
        rng = np.random.default_rng(19)
        embeddings = rng.normal(size=(80, 12))
        private_ids = np.array([f"private-identity-{index // 2}" for index in range(80)])
        config = EmbeddingDiagnosticsConfig(
            spectrum_max_samples=31,
            pairwise_max_samples=29,
            hubness_max_samples=23,
            hubness_k=5,
            seed=7,
        )

        first = compute_embedding_diagnostics(
            embeddings,
            identity_ids=private_ids,
            repeat_ids=private_ids,
            config=config,
        )
        second = compute_embedding_diagnostics(
            embeddings,
            identity_ids=private_ids,
            repeat_ids=private_ids,
            config=config,
        )

        self.assertEqual(first, second)
        self.assertEqual(
            first["schema_version"], EMBEDDING_DIAGNOSTICS_SCHEMA_VERSION
        )
        self.assertEqual(
            first["centered_covariance_spectrum"]["sample_count"], 31
        )
        self.assertEqual(first["hubness"]["sample_count"], 23)
        serialized = json.dumps(first, allow_nan=False, sort_keys=True)
        self.assertNotIn("private-identity", serialized)


if __name__ == "__main__":
    unittest.main()

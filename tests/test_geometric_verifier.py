from __future__ import annotations

import json
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from cvi.geometric_verifier import (
    GeometricAuditCapacityExceeded,
    GeometricCandidatePair,
    GeometricDecision,
    GeometricImageBinding,
    GeometricReason,
    GeometricVerifierPolicy,
    GeometricVerifierRequest,
    canonical_rgb_sha256,
    publish_geometric_evidence,
    verify_geometric_request,
    _apply_d4,
    _geometry_metrics,
    _transform_nondegenerate,
)
from cvi.provenance import content_sha256

try:
    import cv2
    import numpy as np

    BACKEND_AVAILABLE = True
except ImportError:
    BACKEND_AVAILABLE = False

BACKEND_SUPPORTED = (
    BACKEND_AVAILABLE
    and cv2.__version__ == GeometricVerifierPolicy().opencv_reference_version
    and np.__version__ == GeometricVerifierPolicy().numpy_reference_version
)


def _token(value: int) -> str:
    return f"{value:064x}"


def _request(images: list[object], *, d4: tuple[str, ...] = ("ORIGINAL",)) -> GeometricVerifierRequest:
    bindings = tuple(
        sorted(
            (
                GeometricImageBinding(
                    opaque_sample_id=_token(index + 1),
                    canonical_width=image.shape[1],
                    canonical_height=image.shape[0],
                    pixel_sha256=canonical_rgb_sha256(image),
                )
                for index, image in enumerate(images)
            ),
            key=lambda item: item.opaque_sample_id,
        )
    )
    return GeometricVerifierRequest(
        candidates=(GeometricCandidatePair(
            left_opaque_sample_id=_token(1),
            right_opaque_sample_id=_token(2),
            candidate_channels=("PDQ",),
            candidate_evidence_tokens=(_token(100),),
            right_d4_hypotheses=d4,
        ),),
        images=bindings,
        evidence_bindings=(
            ("image_content_receipts_sha256", _token(199)),
            ("pdq_candidates_sha256", _token(200)),
        ),
    )


class GeometricVerifierContractTests(unittest.TestCase):
    def test_policy_config_round_trip_and_initialization_warning(self) -> None:
        path = Path("configs/public_canine_geometric_verifier_policy.example.json")
        policy = GeometricVerifierPolicy.from_dict(json.loads(path.read_text()))
        self.assertEqual(policy, GeometricVerifierPolicy())
        self.assertEqual(policy.threshold_status, "INITIALIZATION_ONLY_NOT_CALIBRATED")
        self.assertIn("LICENSE_RECEIPT_PENDING", policy.backend_admission_status)

    def test_schema_rejects_labels_unsorted_pairs_and_unknown_channels(self) -> None:
        with self.assertRaisesRegex(TypeError, "unexpected keyword"):
            GeometricCandidatePair(  # type: ignore[call-arg]
                left_opaque_sample_id=_token(1),
                right_opaque_sample_id=_token(2),
                candidate_channels=("PDQ",),
                candidate_evidence_tokens=(_token(3),),
                dog_id="forbidden",
            )
        with self.assertRaisesRegex(ValueError, "distinct and sorted"):
            GeometricCandidatePair(
                left_opaque_sample_id=_token(2),
                right_opaque_sample_id=_token(1),
                candidate_channels=("PDQ",),
                candidate_evidence_tokens=(_token(3),),
            )
        with self.assertRaisesRegex(ValueError, "channels"):
            GeometricCandidatePair(
                left_opaque_sample_id=_token(1),
                right_opaque_sample_id=_token(2),
                candidate_channels=("IDENTITY_LABEL",),
                candidate_evidence_tokens=(_token(3),),
            )
        with self.assertRaisesRegex(ValueError, "exactly match"):
            GeometricVerifierRequest(
                candidates=(GeometricCandidatePair(
                    _token(1), _token(2), ("PDQ",), (_token(3),)
                ),),
                images=(
                    GeometricImageBinding(_token(1), 64, 64, _token(4)),
                    GeometricImageBinding(_token(2), 64, 64, _token(5)),
                ),
                evidence_bindings=(("dog_id", _token(6)),),
            )

    def test_backend_absence_is_unresolved_without_loading_pixels(self) -> None:
        fake = [type("Image", (), {"shape": (64, 64, 3)})(), type("Image", (), {"shape": (64, 64, 3)})()]
        # Binding digests need not be materialized because unavailable backends
        # deliberately avoid decoding any source evidence.
        request = GeometricVerifierRequest(
            candidates=(GeometricCandidatePair(
                _token(1), _token(2), ("PHASH",), (_token(3),)
            ),),
            images=(
                GeometricImageBinding(_token(1), 64, 64, _token(4)),
                GeometricImageBinding(_token(2), 64, 64, _token(5)),
            ),
            evidence_bindings=(
                ("image_content_receipts_sha256", _token(5)),
                ("phash_candidates_sha256", _token(6)),
            ),
        )
        with patch("cvi.geometric_verifier._load_backend", return_value=None):
            evidence = verify_geometric_request(
                request, image_loader=lambda _: self.fail("loader must not run")
            )
        result = evidence.results[0]
        self.assertEqual(result.decision, GeometricDecision.UNRESOLVED)
        self.assertEqual(result.reason, GeometricReason.UNRESOLVED_BACKEND_UNAVAILABLE)
        self.assertEqual(dict(evidence.backend)["availability"], "UNAVAILABLE")

    def test_backend_version_mismatch_is_unresolved_without_loading_pixels(self) -> None:
        request = GeometricVerifierRequest(
            candidates=(GeometricCandidatePair(
                _token(1), _token(2), ("PHASH",), (_token(3),)
            ),),
            images=(
                GeometricImageBinding(_token(1), 64, 64, _token(4)),
                GeometricImageBinding(_token(2), 64, 64, _token(5)),
            ),
            evidence_bindings=(
                ("image_content_receipts_sha256", _token(5)),
                ("phash_candidates_sha256", _token(6)),
            ),
        )
        fake_numpy = type("FakeNumpy", (), {"__version__": "999"})()
        fake_cv2 = type("FakeCv2", (), {"__version__": "999"})()
        with patch(
            "cvi.geometric_verifier._load_backend",
            return_value=(fake_numpy, fake_cv2),
        ):
            evidence = verify_geometric_request(
                request, image_loader=lambda _: self.fail("loader must not run")
            )
        self.assertEqual(
            evidence.results[0].reason,
            GeometricReason.UNRESOLVED_BACKEND_VERSION_MISMATCH,
        )
        self.assertEqual(dict(evidence.backend)["availability"], "UNSUPPORTED_VERSION")

    @unittest.skipUnless(
        BACKEND_SUPPORTED, "frozen OpenCV/NumPy reference backend unavailable"
    )
    def test_pixel_binding_mismatch_fails_before_a_decision(self) -> None:
        images = [np.zeros((96, 96, 3), np.uint8), np.zeros((96, 96, 3), np.uint8)]
        request = _request(images)
        forged = replace(
            request,
            images=(replace(request.images[0], pixel_sha256=_token(999)), request.images[1]),
        )
        with self.assertRaisesRegex(ValueError, "digest differs"):
            verify_geometric_request(forged, image_loader=lambda token: images[int(token, 16) - 1])

    def test_caps_fail_before_loader_and_nondefault_policy_is_refused(self) -> None:
        request = GeometricVerifierRequest(
            candidates=(GeometricCandidatePair(_token(1), _token(2), ("PDQ",), (_token(3),)),),
            images=(
                GeometricImageBinding(_token(1), 8192, 8192, _token(4)),
                GeometricImageBinding(_token(2), 8192, 8192, _token(5)),
            ),
            evidence_bindings=(
                ("image_content_receipts_sha256", _token(5)),
                ("pdq_candidates_sha256", _token(6)),
            ),
        )
        with self.assertRaisesRegex(GeometricAuditCapacityExceeded, "image-pixel"):
            verify_geometric_request(request, image_loader=lambda _: self.fail("loader must not run"))
        with self.assertRaisesRegex(ValueError, "frozen initialization"):
            verify_geometric_request(
                request,
                image_loader=lambda _: None,
                policy=replace(GeometricVerifierPolicy(), confirmation_minimum_ssim=0.73),
            )

    def test_no_overwrite_and_content_binding(self) -> None:
        request = GeometricVerifierRequest(
            candidates=(GeometricCandidatePair(_token(1), _token(2), ("PDQ",), (_token(3),)),),
            images=(
                GeometricImageBinding(_token(1), 64, 64, _token(4)),
                GeometricImageBinding(_token(2), 64, 64, _token(5)),
            ),
            evidence_bindings=(
                ("image_content_receipts_sha256", _token(5)),
                ("pdq_candidates_sha256", _token(6)),
            ),
        )
        with patch("cvi.geometric_verifier._load_backend", return_value=None):
            evidence = verify_geometric_request(request, image_loader=lambda _: None)
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "evidence.json"
            digest = publish_geometric_evidence(
                output, evidence, tool_provenance={"source": "synthetic-test"}
            )
            payload = json.loads(output.read_text())
            self.assertEqual(payload["bundle_sha256"], digest)
            unsigned = dict(payload)
            del unsigned["bundle_sha256"]
            self.assertEqual(content_sha256(unsigned), digest)
            self.assertNotIn("dog_id", output.read_text())
            with self.assertRaises(FileExistsError):
                publish_geometric_evidence(
                    output, evidence, tool_provenance={"source": "synthetic-test"}
                )

    def test_pair_and_count_tampering_is_rejected(self) -> None:
        request = GeometricVerifierRequest(
            candidates=(GeometricCandidatePair(
                _token(1), _token(2), ("PDQ",), (_token(3),)
            ),),
            images=(
                GeometricImageBinding(_token(1), 64, 64, _token(4)),
                GeometricImageBinding(_token(2), 64, 64, _token(5)),
            ),
            evidence_bindings=(
                ("image_content_receipts_sha256", _token(5)),
                ("pdq_candidates_sha256", _token(6)),
            ),
        )
        with patch("cvi.geometric_verifier._load_backend", return_value=None):
            evidence = verify_geometric_request(request, image_loader=lambda _: None)
        result = evidence.results[0]
        with self.assertRaisesRegex(ValueError, "decision/reason"):
            replace(result, decision=GeometricDecision.GEOMETRIC_CONFIRMED)
        with self.assertRaisesRegex(ValueError, "token differs"):
            replace(result, evidence_token=_token(999))
        with self.assertRaisesRegex(ValueError, "canonical"):
            replace(
                result,
                candidate_evidence_tokens=(_token(4), _token(3)),
            )
        with self.assertRaisesRegex(ValueError, "counts differ"):
            replace(evidence, counts=(("UNRESOLVED", 1),))

    @unittest.skipUnless(BACKEND_AVAILABLE, "optional numerical backend unavailable")
    def test_pdq_d4_coordinate_mapping_golden(self) -> None:
        values = np.arange(6, dtype=np.uint8).reshape(2, 3)
        image = np.repeat(values[:, :, None], 3, axis=2)
        expected = {
            "ORIGINAL": [[0, 1, 2], [3, 4, 5]],
            "ROT90CCW": [[2, 5], [1, 4], [0, 3]],
            "ROT180": [[5, 4, 3], [2, 1, 0]],
            "ROT270CCW": [[3, 0], [4, 1], [5, 2]],
            "FLIP_X": [[3, 4, 5], [0, 1, 2]],
            "FLIP_Y": [[2, 1, 0], [5, 4, 3]],
            "FLIP_PLUS_DIAGONAL": [[0, 3], [1, 4], [2, 5]],
            "FLIP_MINUS_DIAGONAL": [[5, 2], [4, 1], [3, 0]],
        }
        for name, raster in expected.items():
            with self.subTest(name=name):
                self.assertEqual(_apply_d4(image, name, np)[:, :, 0].tolist(), raster)

    @unittest.skipUnless(BACKEND_AVAILABLE, "optional numerical backend unavailable")
    def test_foldover_and_projective_denominator_crossing_are_degenerate(self) -> None:
        policy = GeometricVerifierPolicy()
        shape = (200, 300)
        identity = np.eye(3, dtype=np.float64)
        reflected = np.asarray(
            [[-1.0, 0.0, 299.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        denominator_crossing = np.asarray(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.01, 0.0, -1.0]],
            dtype=np.float64,
        )
        self.assertTrue(
            _transform_nondegenerate(
                identity, shape, shape, policy, np, cv2
            )[0]
        )
        self.assertFalse(
            _transform_nondegenerate(
                reflected, shape, shape, policy, np, cv2
            )[0]
        )
        self.assertFalse(
            _transform_nondegenerate(
                denominator_crossing, shape, shape, policy, np, cv2
            )[0]
        )

    @unittest.skipUnless(BACKEND_AVAILABLE, "optional numerical backend unavailable")
    def test_p95_uses_all_estimator_inliers_before_error_gate(self) -> None:
        source = np.asarray(
            [
                [10, 10], [50, 10], [90, 10], [130, 10],
                [10, 50], [50, 50], [90, 50], [130, 50],
                [10, 90], [50, 90], [90, 90], [130, 90],
            ],
            dtype=np.float64,
        )
        target = source.copy()
        target[-1] += (45.0, 0.0)
        metrics = _geometry_metrics(
            source,
            target,
            np.ones((len(source), 1), dtype=np.uint8),
            np.eye(3, dtype=np.float64),
            (120, 180),
            (120, 180),
            GeometricVerifierPolicy(),
            np,
            cv2,
        )
        self.assertIsNotNone(metrics)
        assert metrics is not None
        self.assertEqual(metrics["inlier_count"], len(source))
        self.assertGreater(
            metrics["p95_symmetric_error_fraction"],
            GeometricVerifierPolicy().maximum_p95_symmetric_error_fraction,
        )
        self.assertFalse(metrics["spatial_pass"])


@unittest.skipUnless(
    BACKEND_SUPPORTED, "frozen OpenCV/NumPy reference backend unavailable"
)
class GeometricVerifierSyntheticTests(unittest.TestCase):
    @staticmethod
    def _texture(seed: int = 7) -> object:
        generator = np.random.default_rng(seed)
        image = generator.integers(0, 256, size=(420, 520, 3), dtype=np.uint8)
        image = cv2.GaussianBlur(image, (5, 5), 0.8)
        for index in range(40):
            center = (30 + (index * 83) % 450, 30 + (index * 61) % 350)
            cv2.circle(image, center, 5 + index % 19, ((index * 47) % 256, (index * 89) % 256, (index * 131) % 256), 2)
        return image

    def test_affine_duplicate_confirms_deterministically(self) -> None:
        left = self._texture()
        matrix = cv2.getRotationMatrix2D((260, 210), 3.0, 0.98)
        matrix[:, 2] += (8, -5)
        right = cv2.warpAffine(left, matrix, (520, 420), borderMode=cv2.BORDER_REFLECT)
        images = [left, right]
        request = _request(images)
        loader = lambda token: images[int(token, 16) - 1]
        first = verify_geometric_request(request, image_loader=loader)
        second = verify_geometric_request(request, image_loader=loader)
        self.assertEqual(first, second)
        self.assertEqual(first.results[0].decision, GeometricDecision.GEOMETRIC_CONFIRMED)
        self.assertIn(first.results[0].selected_model, {"PARTIAL_AFFINE", "AFFINE", "HOMOGRAPHY_USAC_MAGSAC"})

    def test_low_texture_is_unresolved_not_rejected(self) -> None:
        images = [np.full((200, 240, 3), 127, np.uint8), np.full((200, 240, 3), 128, np.uint8)]
        evidence = verify_geometric_request(
            _request(images), image_loader=lambda token: images[int(token, 16) - 1]
        )
        self.assertEqual(evidence.results[0].decision, GeometricDecision.UNRESOLVED)
        self.assertIn(evidence.results[0].reason, {
            GeometricReason.UNRESOLVED_LOW_TEXTURE,
            GeometricReason.UNRESOLVED_INSUFFICIENT_MUTUAL_MATCHES,
        })

    def test_supported_geometry_with_clear_full_overlap_conflict_rejects(self) -> None:
        def background(seed: int) -> object:
            generator = np.random.default_rng(seed)
            image = generator.integers(
                0, 256, size=(420, 520, 3), dtype=np.uint8
            )
            image = cv2.GaussianBlur(image, (5, 5), 0.8)
            # Shared, spatially distributed landmarks establish a valid
            # non-degenerate transform while the complete overlap remains
            # independently generated.  This exercises rejection evidence,
            # not a claim that real thresholds are calibrated.
            for index in range(25):
                center = (
                    30 + (index * 83) % 460,
                    30 + (index * 61) % 360,
                )
                color = (
                    (index * 47) % 256,
                    (index * 89) % 256,
                    (index * 131) % 256,
                )
                cv2.circle(image, center, 7 + index % 11, color, 3)
                cv2.line(
                    image,
                    (center[0] - 10, center[1]),
                    (center[0] + 10, center[1]),
                    color,
                    2,
                )
            return image

        images = [background(7), background(99)]
        evidence = verify_geometric_request(
            _request(images),
            image_loader=lambda token: images[int(token, 16) - 1],
        )
        result = evidence.results[0]
        self.assertEqual(result.decision, GeometricDecision.GEOMETRIC_REJECTED)
        self.assertEqual(
            result.reason,
            GeometricReason.REJECTED_PHOTOMETRIC_CONTRADICTION,
        )

    def test_declared_horizontal_flip_can_confirm(self) -> None:
        left = self._texture(19)
        right = np.ascontiguousarray(np.flip(left, axis=1))
        images = [left, right]
        request = _request(images, d4=("ORIGINAL", "FLIP_Y"))
        evidence = verify_geometric_request(
            request, image_loader=lambda token: images[int(token, 16) - 1]
        )
        self.assertEqual(evidence.results[0].decision, GeometricDecision.GEOMETRIC_CONFIRMED)
        self.assertEqual(evidence.results[0].selected_right_d4, "FLIP_Y")


if __name__ == "__main__":
    unittest.main()

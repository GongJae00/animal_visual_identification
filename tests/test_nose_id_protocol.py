from __future__ import annotations

from dataclasses import replace
from pathlib import PurePosixPath
import unittest
import uuid

import numpy as np

from embedding.methods.nose.data.dataset import NoseIDSample
from embedding.methods.nose.data.protocol import (
    build_dev_n3_folds,
    capture_id,
    select_temporally_farthest,
    stable_capture_order_key,
)


def _sample(
    identity: str,
    video: str,
    timestamp_ms: int,
    *,
    sample_suffix: str = "",
) -> NoseIDSample:
    return NoseIDSample(
        sample_id=f"{identity}-{video}-{timestamp_ms}-{sample_suffix}",
        image_path=PurePosixPath("image.png"),
        image_sha256="a" * 64,
        image_width=100,
        image_height=100,
        registered_dog_id=identity,
        session_id=f"session-{identity}-{video}",
        camera_id="camera-1",
        video_id=video,
        frame_index=timestamp_ms // 10,
        timestamp_ms=timestamp_ms,
        nose_bbox_xyxy=(1.0, 1.0, 10.0, 10.0),
        keypoints_xy=np.zeros((6, 2), dtype=np.float32),
        keypoint_visibility=np.full(6, 2, dtype=np.int64),
        semantic_mask_path=PurePosixPath("semantic.png"),
        semantic_mask_sha256="b" * 64,
        semantic_mask_box_xyxy=(1.0, 1.0, 10.0, 10.0),
        invalid_mask_path=PurePosixPath("invalid.png"),
        invalid_mask_sha256="c" * 64,
        invalid_mask_box_xyxy=(1.0, 1.0, 10.0, 10.0),
        split_role="DEV",
    )


class NoseIDProtocolTests(unittest.TestCase):
    def test_folds_are_stable_capture_disjoint_and_use_every_other_capture(self) -> None:
        identities = [
            str(uuid.uuid5(uuid.NAMESPACE_DNS, "dog-a")),
            str(uuid.uuid5(uuid.NAMESPACE_DNS, "dog-b")),
        ]
        rows = [
            _sample(identity, video, timestamp)
            for identity in identities
            for video in ("v1", "v2", "v3", "v4")
            for timestamp in (0, 250, 500, 750, 1000, 1500)
        ]
        folds = build_dev_n3_folds(rows, seed=23)
        reversed_folds = build_dev_n3_folds(list(reversed(rows)), seed=23)

        self.assertEqual(folds, reversed_folds)
        self.assertEqual(len(folds), 3)
        two_capture_minimum = [
            row
            for row in rows
            if row.registered_dog_id == identities[0] or row.video_id in {"v1", "v2"}
        ]
        self.assertEqual(len(build_dev_n3_folds(two_capture_minimum, seed=23)), 2)
        expected_gallery = {
            identity: sorted(
                {
                    capture_id(row)
                    for row in rows
                    if row.registered_dog_id == identity
                },
                key=lambda value: (stable_capture_order_key(23, identity, value), value),
            )[:3]
            for identity in identities
        }
        for fold_index, fold in enumerate(folds):
            self.assertEqual(len(fold.gallery), 2)
            self.assertEqual(len(fold.queries), 6)
            self.assertFalse(
                {template.capture_id for template in fold.gallery}
                & {template.capture_id for template in fold.queries}
            )
            for gallery in fold.gallery:
                self.assertEqual(
                    gallery.capture_id,
                    expected_gallery[gallery.identity_id][fold_index],
                )
                self.assertEqual(len(gallery.samples), 4)
                query_captures = {
                    template.capture_id
                    for template in fold.queries
                    if template.identity_id == gallery.identity_id
                }
                self.assertEqual(len(query_captures), 3)
                self.assertNotIn(gallery.capture_id, query_captures)

    def test_temporal_selection_is_farthest_deterministic_and_quality_free(self) -> None:
        identity = str(uuid.uuid5(uuid.NAMESPACE_DNS, "dog-a"))
        rows = [
            _sample(identity, "v1", value)
            for value in (0, 100, 500, 900, 1400, 2000)
        ]
        selected = select_temporally_farthest(list(reversed(rows)))
        self.assertEqual([sample.timestamp_ms for sample in selected], [0, 900, 1400, 2000])
        self.assertTrue(
            all(
                later.timestamp_ms - earlier.timestamp_ms >= 400
                for earlier, later in zip(selected, selected[1:])
            )
        )

    def test_protocol_rejects_non_dev_and_identities_without_query_capture(self) -> None:
        identity = str(uuid.uuid5(uuid.NAMESPACE_DNS, "dog-a"))
        only_capture = [_sample(identity, "v1", 0)]
        with self.assertRaisesRegex(ValueError, "at least two captures"):
            build_dev_n3_folds(only_capture, seed=0)
        with self.assertRaisesRegex(ValueError, "only DEV"):
            build_dev_n3_folds(
                [replace(only_capture[0], split_role="TRAIN")], seed=0
            )


if __name__ == "__main__":
    unittest.main()

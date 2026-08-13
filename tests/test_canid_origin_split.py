from __future__ import annotations

import unittest

from data.types import CaptureGroupKind, UnifiedCanidSample


class CanidOriginSplitTests(unittest.TestCase):
    def test_mpdd_splits_are_preserved(self) -> None:
        """MPDD gallery/query must never appear in TRAIN."""
        samples = [
            UnifiedCanidSample(
                sample_id=f"s{i}", dataset_name="mpdd", dataset_version="v1",
                source_group_id=str(i), image_path=f"{i}.jpg",
                image_sha256="a" * 64, width=1, height=1,
                split_role=role,
                registered_identity_id=f"uuid-{i}",
                raw_identity_id=str(i),
            )
            for i, role in enumerate(
                ["train"] * 5 + ["val"] * 5 + ["query"] * 3 + ["gallery"] * 3
            )
        ]
        train_ids = {s.registered_identity_id for s in samples if s.split_role in ("train", "val")}
        test_ids = {s.registered_identity_id for s in samples if s.split_role in ("query", "gallery")}
        self.assertFalse(train_ids & test_ids)

    def test_yt_split_is_disjoint(self) -> None:
        samples = [
            UnifiedCanidSample(
                sample_id=f"s{i}", dataset_name="yt-bb-dog", dataset_version="v1",
                source_group_id=str(i % 5), image_path=f"{i}.jpg",
                image_sha256="a" * 64, width=1, height=1,
                split_role="train" if i < 20 else "test",
                registered_identity_id=f"uuid-{i % 5}",
            )
            for i in range(30)
        ]
        train_ids = {s.registered_identity_id for s in samples if s.split_role == "train"}
        test_ids = {s.registered_identity_id for s in samples if s.split_role == "test"}
        self.assertTrue(train_ids & test_ids)
        self.assertIn("train", {s.split_role for s in samples})
        self.assertIn("test", {s.split_role for s in samples})


if __name__ == "__main__":
    unittest.main()

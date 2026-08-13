from __future__ import annotations

import unittest

from data.label_map import (
    CANID_KEYPOINT_ALIASES,
    CANID_SPECIES,
    is_known_canid_species,
    resolve_keypoint_name,
)


class CanidLabelMapTests(unittest.TestCase):
    def test_resolves_canonical_names(self) -> None:
        self.assertEqual(resolve_keypoint_name("nose_tip"), "nose_center")
        self.assertEqual(resolve_keypoint_name("left_eye"), "left_eye")
        self.assertEqual(resolve_keypoint_name("leye"), "left_eye")
        self.assertIsNone(resolve_keypoint_name("unknown_keypoint"))

    def test_case_and_separator_normalization(self) -> None:
        self.assertEqual(resolve_keypoint_name("Left Eye"), "left_eye")
        self.assertEqual(resolve_keypoint_name("LEFT_EYE"), "left_eye")
        self.assertEqual(resolve_keypoint_name("left-eye"), "left_eye")

    def test_known_species_detection(self) -> None:
        self.assertTrue(is_known_canid_species("Canis lupus familiaris"))
        self.assertTrue(is_known_canid_species("Vulpes vulpes"))
        self.assertFalse(is_known_canid_species("Felis catus"))

    def test_alias_map_includes_all_canonical_names(self) -> None:
        for canonical, aliases in CANID_KEYPOINT_ALIASES.items():
            self.assertIn(canonical.lower(), [a.lower() for a in aliases] or [canonical.lower()])

    def test_species_list_is_nonempty(self) -> None:
        self.assertGreater(len(CANID_SPECIES), 0)


if __name__ == "__main__":
    unittest.main()

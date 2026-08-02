from __future__ import annotations

import os
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from data_pipeline.acquisition import sha256_file
from foundation.provenance import content_sha256
from artifact_contracts.runtime_library_provenance import (
    ExpectedRuntimeBinary,
    RuntimeLibraryManifest,
    RuntimeLibraryPhase,
    RuntimeLibraryPolicy,
    RuntimeLibraryTracker,
    freeze_runtime_library_policy,
    parse_executable_mappings,
)


def maps_line(path: Path, executable: bool = True) -> bytes:
    info = path.stat()
    device = f"{os.major(info.st_dev):02x}:{os.minor(info.st_dev):02x}"
    permissions = "r-xp" if executable else "r--p"
    return (
        f"00001000-00002000 {permissions} 00000000 {device} "
        f"{info.st_ino} {path}\n"
    ).encode()


class RuntimeLibraryProvenanceTests(unittest.TestCase):
    def test_policy_and_manifest_roundtrip_are_strict(self) -> None:
        with TemporaryDirectory() as temporary:
            binary = Path(temporary) / "runtime.bin"
            binary.write_bytes(b"runtime-binary")
            expected = ExpectedRuntimeBinary(
                resolved_path=str(binary.resolve()),
                byte_size=binary.stat().st_size,
                content_sha256=sha256_file(binary),
            )
            expected_set_sha256 = content_sha256([
                (
                    expected.resolved_path,
                    expected.byte_size,
                    expected.content_sha256,
                )
            ])
            policy = RuntimeLibraryPolicy(
                expected_binaries=(expected,),
                discovery_binary_set_sha256=expected_set_sha256,
            )
            self.assertEqual(
                RuntimeLibraryPolicy.from_dict(policy.to_dict()),
                policy,
            )

            maps = Path(temporary) / "maps"
            maps.write_bytes(maps_line(binary))
            tracker = RuntimeLibraryTracker(policy, maps_path=maps)
            for phase in RuntimeLibraryPhase:
                tracker.capture(phase)
            manifest = tracker.finalize()
            self.assertEqual(
                RuntimeLibraryManifest.from_dict(manifest.to_dict()),
                manifest,
            )
            forged = manifest.to_dict()
            forged["binary_set_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "set hash"):
                RuntimeLibraryManifest.from_dict(forged)

    def test_parser_groups_segments_and_rejects_anonymous_deleted_aliases(self) -> None:
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "library with spaces.so"
            path.write_bytes(b"runtime-binary")
            payload = maps_line(path) + maps_line(path) + (
                b"00003000-00004000 r-xp 00000000 00:00 0 [vdso]\n"
                b"ffffffffff600000-ffffffffff601000 --xp 00000000 "
                b"00:00 0 [vsyscall]\n"
            )
            parsed = parse_executable_mappings(payload, maximum_lines=10)
            self.assertEqual(len(parsed), 1)
            self.assertEqual(parsed[0][3], str(path))
            with self.assertRaisesRegex(ValueError, "anonymous executable"):
                parse_executable_mappings(
                    b"00001000-00002000 r-xp 00000000 00:00 0\n",
                    maximum_lines=10,
                )
            with self.assertRaisesRegex(ValueError, "special"):
                parse_executable_mappings(
                    b"00003000-00004000 r-xp 00000000 00:00 0 [anonymous]\n",
                    maximum_lines=10,
                )
            with self.assertRaisesRegex(ValueError, "deleted executable"):
                parse_executable_mappings(
                    maps_line(path).rstrip() + b" (deleted)\n",
                    maximum_lines=10,
                )

    def test_discovery_tracks_phases_and_hashes_one_binary_once(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            binary = root / "runtime.so"
            binary.write_bytes(b"runtime-binary")
            maps = root / "maps"
            maps.write_bytes(maps_line(binary))
            policy = RuntimeLibraryPolicy(
                expected_binaries=(),
                allow_discovery_only=True,
                maximum_maps_bytes=10_000,
                maximum_maps_lines=100,
                maximum_executable_identities=10,
                maximum_individual_binary_bytes=1_000,
                maximum_total_binary_bytes=10_000,
                hash_chunk_bytes=3,
            )
            tracker = RuntimeLibraryTracker(policy, maps)
            for phase in RuntimeLibraryPhase:
                tracker.capture(phase)
            manifest = tracker.finalize()
            self.assertEqual(manifest.decision, "DISCOVERY_ONLY")
            self.assertEqual(manifest.binary_bytes_hashed, binary.stat().st_size)
            self.assertEqual(manifest.maps_snapshots, 5)
            self.assertEqual(manifest.entries[0].first_seen_phase, RuntimeLibraryPhase.DEPENDENCIES_IMPORTED)
            self.assertEqual(manifest.entries[0].last_seen_phase, RuntimeLibraryPhase.FINAL_OUTPUT_READY)
            strict = freeze_runtime_library_policy(policy, (manifest, manifest))
            self.assertFalse(strict.allow_discovery_only)
            self.assertEqual(
                strict.discovery_binary_set_sha256,
                manifest.binary_set_sha256,
            )
            strict_tracker = RuntimeLibraryTracker(strict, maps)
            for phase in RuntimeLibraryPhase:
                strict_tracker.capture(phase)
            self.assertEqual(strict_tracker.finalize().decision, "PASS")

    def test_strict_expected_set_passes_and_path_or_hash_substitution_fails(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            binary = root / "runtime.so"
            binary.write_bytes(b"runtime-binary")
            maps = root / "maps"
            maps.write_bytes(maps_line(binary))
            expected = ExpectedRuntimeBinary(
                resolved_path=str(binary),
                byte_size=binary.stat().st_size,
                content_sha256=sha256_file(binary),
            )
            base = RuntimeLibraryPolicy(
                expected_binaries=(expected,),
                discovery_binary_set_sha256=content_sha256([
                    (
                        expected.resolved_path,
                        expected.byte_size,
                        expected.content_sha256,
                    )
                ]),
                maximum_maps_bytes=10_000,
                maximum_maps_lines=100,
                maximum_executable_identities=10,
                maximum_individual_binary_bytes=1_000,
                maximum_total_binary_bytes=10_000,
                hash_chunk_bytes=4,
            )
            tracker = RuntimeLibraryTracker(base, maps)
            for phase in RuntimeLibraryPhase:
                tracker.capture(phase)
            self.assertEqual(tracker.finalize().decision, "PASS")
            substituted = replace(expected, content_sha256="0" * 64)
            substituted_policy = RuntimeLibraryPolicy(
                expected_binaries=(substituted,),
                discovery_binary_set_sha256=content_sha256([
                    (
                        substituted.resolved_path,
                        substituted.byte_size,
                        substituted.content_sha256,
                    )
                ]),
                maximum_maps_bytes=10_000,
                maximum_maps_lines=100,
                maximum_executable_identities=10,
                maximum_individual_binary_bytes=1_000,
                maximum_total_binary_bytes=10_000,
                hash_chunk_bytes=4,
            )
            tracker = RuntimeLibraryTracker(substituted_policy, maps)
            for phase in RuntimeLibraryPhase:
                tracker.capture(phase)
            self.assertEqual(tracker.finalize().decision, "FAIL")

    def test_phase_order_and_resource_caps_fail_closed(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            binary = root / "runtime.so"
            binary.write_bytes(b"runtime-binary")
            maps = root / "maps"
            maps.write_bytes(maps_line(binary))
            policy = RuntimeLibraryPolicy(
                expected_binaries=(),
                allow_discovery_only=True,
                maximum_maps_bytes=1,
            )
            tracker = RuntimeLibraryTracker(policy, maps)
            with self.assertRaisesRegex(ValueError, "maps exceed"):
                tracker.capture(RuntimeLibraryPhase.DEPENDENCIES_IMPORTED)
            normal = replace(policy, maximum_maps_bytes=10_000)
            tracker = RuntimeLibraryTracker(normal, maps)
            with self.assertRaisesRegex(ValueError, "exactly once in order"):
                tracker.capture(RuntimeLibraryPhase.SESSION_READY)


if __name__ == "__main__":
    unittest.main()

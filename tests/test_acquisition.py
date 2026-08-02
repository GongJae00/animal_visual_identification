from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from data_pipeline.acquisition import (
    AcquisitionManifest,
    CameraSpecification,
    IRMechanism,
    ModalityInterval,
    ModalityState,
    RawVideoRecord,
    TimestampAuditAccumulator,
    VideoProbeSummary,
    audit_timestamp_lines,
    parse_ffprobe,
    sha256_file,
)


class AcquisitionTests(unittest.TestCase):
    def complete_spec(
        self,
        camera_id: str,
        version: str = "settings-v1",
    ) -> CameraSpecification:
        return CameraSpecification(
            camera_id=camera_id,
            camera_setting_version=version,
            sensor_model="sensor-x",
            ir_mechanism=IRMechanism.DAY_NIGHT_SWITCHING,
            ir_spectral_band="850nm NIR",
            width=1920,
            height=1080,
            stored_fps=30.0,
            shutter="rolling",
            gain_mode="auto, logged when available",
            exposure_mode="auto, logged when available",
            white_balance_mode="auto",
            wdr_enabled=True,
            ir_cut_behavior="automatic day/night switch",
            codec="h264",
            target_bitrate_mbps=4.0,
            gop_length=30,
            focus_mode="fixed",
            focal_length_mm=3.6,
            horizontal_fov_deg=90.0,
            installation_height_m=1.8,
            cage_center_distance_m=1.5,
            pan_deg=0.0,
            tilt_deg=-20.0,
            timestamp_accuracy_ms=20.0,
            measured_frame_drop_rate=0.001,
        )

    def test_incomplete_camera_spec_reports_exact_missing_fields(self) -> None:
        spec = CameraSpecification(
            camera_id="cam-01",
            camera_setting_version="settings-v1",
            sensor_model="sensor-x",
            width=1920,
            height=1080,
            stored_fps=29.97,
        )
        self.assertFalse(spec.g0_ready)
        self.assertIn("ir_mechanism", spec.missing_for_g0)
        self.assertIn("codec", spec.missing_for_g0)
        self.assertNotIn("width", spec.missing_for_g0)

    def test_ffprobe_fixture_is_parsed_without_nominal_fps_rounding(self) -> None:
        fixture = Path(__file__).parent / "fixtures" / "ffprobe_video.json"
        summary = parse_ffprobe(json.loads(fixture.read_text(encoding="utf-8")))
        self.assertEqual(summary.codec, "h264")
        self.assertEqual(summary.width, 1920)
        self.assertEqual(summary.height, 1080)
        self.assertAlmostEqual(summary.average_fps, 30000 / 1001)
        self.assertEqual(summary.frame_count, 1800)
        self.assertEqual(summary.bitrate_bps, 4_000_000)

    def test_raw_record_requires_sorted_nonoverlapping_modalities(self) -> None:
        fixture = Path(__file__).parent / "fixtures" / "ffprobe_video.json"
        probe = parse_ffprobe(json.loads(fixture.read_text(encoding="utf-8")))
        with self.assertRaisesRegex(ValueError, "overlap"):
            RawVideoRecord(
                source_id="source-01",
                source_uri="/protected/source.mp4",
                source_sha256="a" * 64,
                byte_size=100,
                camera_id="cam-01",
                cage_id="cage-01",
                camera_setting_version="settings-v1",
                recording_start_ns=1_000_000_000,
                recording_end_ns=61_060_000_000,
                probe=probe,
                modality_intervals=(
                    ModalityInterval(
                        1_000_000_000,
                        41_000_000_000,
                        ModalityState.RGB,
                    ),
                    ModalityInterval(
                        40_000_000_000,
                        61_060_000_000,
                        ModalityState.TRANSITION,
                    ),
                ),
            )

    def test_manifest_gate_accepts_three_complete_day_night_cages(self) -> None:
        duration_seconds = 86_400.0
        probe = VideoProbeSummary(
            codec="h264",
            format_name="matroska",
            width=1920,
            height=1080,
            average_fps=30.0,
            duration_seconds=duration_seconds,
            bitrate_bps=4_000_000,
            frame_count=2_592_000,
            time_base="1/90000",
        )
        duration_ns = round(duration_seconds * 1_000_000_000)
        cameras = tuple(self.complete_spec(f"cam-{index}") for index in range(3))
        videos = tuple(
            RawVideoRecord(
                source_id=f"source-{index}",
                source_uri=f"/protected/source-{index}.mkv",
                source_sha256=f"{index + 1:064x}",
                byte_size=43_200_000_000,
                camera_id=f"cam-{index}",
                cage_id=f"cage-{index}",
                camera_setting_version="settings-v1",
                recording_start_ns=0,
                recording_end_ns=duration_ns,
                probe=probe,
                modality_intervals=(
                    ModalityInterval(0, duration_ns // 2, ModalityState.RGB),
                    ModalityInterval(
                        duration_ns // 2,
                        duration_ns // 2 + 1_000_000_000,
                        ModalityState.TRANSITION,
                    ),
                    ModalityInterval(
                        duration_ns // 2 + 1_000_000_000,
                        duration_ns,
                        ModalityState.IR,
                    ),
                ),
            )
            for index in range(3)
        )
        manifest = AcquisitionManifest(cameras=cameras, videos=videos)
        self.assertEqual(manifest.gate_blockers(), ())
        self.assertEqual(len(manifest.manifest_sha256), 64)
        round_tripped = AcquisitionManifest.from_dict(
            json.loads(json.dumps(manifest.to_dict()))
        )
        self.assertEqual(round_tripped.to_dict(), manifest.to_dict())
        self.assertEqual(round_tripped.manifest_sha256, manifest.manifest_sha256)

    def test_manifest_rejects_duplicate_source_content(self) -> None:
        probe = VideoProbeSummary(
            codec="h264",
            format_name="mp4",
            width=320,
            height=240,
            average_fps=10.0,
            duration_seconds=1.0,
            bitrate_bps=10_000,
            frame_count=10,
            time_base="1/10240",
        )

        def video(source_id: str) -> RawVideoRecord:
            return RawVideoRecord(
                source_id=source_id,
                source_uri=f"/protected/{source_id}.mp4",
                source_sha256="a" * 64,
                byte_size=100,
                camera_id="cam-01",
                cage_id="cage-01",
                camera_setting_version="settings-v1",
                recording_start_ns=0,
                recording_end_ns=1_000_000_000,
                probe=probe,
                modality_intervals=(
                    ModalityInterval(0, 1_000_000_000, ModalityState.RGB),
                ),
            )

        with self.assertRaisesRegex(ValueError, "source_sha256"):
            AcquisitionManifest(
                cameras=(self.complete_spec("cam-01"),),
                videos=(video("source-a"), video("source-b")),
            )

    def test_manifest_does_not_pool_modality_coverage_across_cages(self) -> None:
        probe = VideoProbeSummary(
            codec="h264",
            format_name="mp4",
            width=320,
            height=240,
            average_fps=10.0,
            duration_seconds=1.0,
            bitrate_bps=10_000,
            frame_count=10,
            time_base="1/10240",
        )
        states = (
            ModalityState.RGB,
            ModalityState.IR,
            ModalityState.TRANSITION,
        )
        cameras = tuple(self.complete_spec(f"cam-{index}") for index in range(3))
        videos = tuple(
            RawVideoRecord(
                source_id=f"source-{index}",
                source_uri=f"/protected/source-{index}.mp4",
                source_sha256=f"{index + 10:064x}",
                byte_size=100,
                camera_id=f"cam-{index}",
                cage_id=f"cage-{index}",
                camera_setting_version="settings-v1",
                recording_start_ns=0,
                recording_end_ns=1_000_000_000,
                probe=probe,
                modality_intervals=(
                    ModalityInterval(0, 1_000_000_000, states[index]),
                ),
            )
            for index in range(3)
        )
        blockers = AcquisitionManifest(
            cameras=cameras,
            videos=videos,
        ).gate_blockers(
            minimum_cages=3,
            minimum_contiguous_seconds_per_cage=1.0,
        )
        self.assertIn("samples:cage:cage-0:missing_modality:IR", blockers)
        self.assertIn(
            "samples:cage:cage-1:missing_modality:RGB_IR_TRANSITION",
            blockers,
        )
        self.assertIn("samples:cage:cage-2:missing_modality:RGB", blockers)

    def test_manifest_parser_rejects_unknown_fields(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown fields"):
            AcquisitionManifest.from_dict(
                {
                    "schema_version": "cvi.acquisition.v1",
                    "cameras": [],
                    "videos": [],
                    "identity_label": "must-not-enter-acquisition",
                }
            )

    def test_timestamp_audit_is_streaming_and_counts_anomalies(self) -> None:
        audit = TimestampAuditAccumulator(expected_period_ns=10)
        for timestamp in (100, 110, 110, 140, 130, 150):
            audit.observe(timestamp)
        result = audit.snapshot()
        self.assertEqual(result.observed_frames, 6)
        self.assertEqual(result.unavailable_timestamps, 0)
        self.assertEqual(result.first_timestamp_ns, 100)
        self.assertEqual(result.last_timestamp_ns, 150)
        self.assertEqual(result.duplicates, 1)
        self.assertEqual(result.inversions, 1)
        self.assertEqual(result.estimated_missing_frames, 3)
        self.assertEqual(result.maximum_forward_gap_ns, 30)

    def test_timestamp_text_audit_handles_negative_and_unavailable_pts(self) -> None:
        result = audit_timestamp_lines(
            (
                "-0.033333333\n",
                "0.000000000\n",
                "N/A\n",
                "\n",
                "0.066666667,side-data-is-ignored\n",
            ),
            expected_fps=30.0,
        )
        self.assertEqual(result.observed_frames, 3)
        self.assertEqual(result.unavailable_timestamps, 2)
        self.assertEqual(result.first_timestamp_ns, -33_333_333)
        self.assertEqual(result.last_timestamp_ns, 66_666_667)
        self.assertEqual(result.estimated_missing_frames, 1)

    def test_timestamp_text_audit_rejects_malformed_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid ffprobe timestamp"):
            audit_timestamp_lines(("not-a-timestamp\n",), expected_fps=30.0)

    def test_source_hash_uses_file_bytes(self) -> None:
        payload = b"protected-video-placeholder"
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "source.bin"
            source.write_bytes(payload)
            self.assertEqual(
                sha256_file(source, chunk_bytes=3),
                hashlib.sha256(payload).hexdigest(),
            )


if __name__ == "__main__":
    unittest.main()

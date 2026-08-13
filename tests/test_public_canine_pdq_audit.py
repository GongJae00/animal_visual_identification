from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory

from embedding.methods.classical.pdq.contracts import PDQFingerprint, PDQSearchPolicy
from foundation.provenance import content_sha256
from embedding.methods.classical.pdq.public_audit import (
    PDQFingerprintChunk,
    build_pdq_evidence_bundle,
    merge_pdq_fingerprint_chunks,
    publish_pdq_evidence_bundle,
    publish_pdq_fingerprint_chunk,
    read_pdq_fingerprint_chunk,
)


def _token(value: int) -> str:
    return f"{value:064x}"


def _fingerprint(value: int, hash_value: int = 0) -> PDQFingerprint:
    return PDQFingerprint(
        opaque_sample_id=_token(value),
        d4_hashes=(f"{hash_value:064x}",) * 8,
        quality=100,
    )


def _chunk(start: int, end: int) -> PDQFingerprintChunk:
    receipt_bindings = ({"receipt": _token(110)},)
    return PDQFingerprintChunk(
        source_spec_sha256=_token(100),
        source_receipt_bindings_sha256=content_sha256(receipt_bindings),
        native_build_receipt_sha256=_token(102),
        native_binary_sha256=_token(103),
        official_regression_receipt_sha256=_token(104),
        corpus_sample_ids_sha256=content_sha256([_token(1), _token(2), _token(3)]),
        corpus_sample_count=3,
        start_index=start,
        end_index=end,
        fingerprints=tuple(_fingerprint(index + 1, index) for index in range(start, end)),
    )


def _context() -> SimpleNamespace:
    receipt_bindings = ({"receipt": _token(110)},)
    return SimpleNamespace(
        source_spec_sha256=_token(100),
        source_receipt_bindings=receipt_bindings,
        source_receipt_bindings_sha256=content_sha256(receipt_bindings),
        native_build_receipt_sha256=_token(102),
        native_build_receipt=SimpleNamespace(binary_sha256=_token(103)),
        official_regression_receipt_sha256=_token(104),
        corpus_sample_ids_sha256=content_sha256([_token(1), _token(2), _token(3)]),
        items=tuple(SimpleNamespace(opaque_id=_token(index)) for index in range(1, 4)),
    )


class PublicCaninePDQAuditTests(unittest.TestCase):
    def test_exact_merge_coverage_and_bounded_mih_publication(self) -> None:
        context = _context()
        manifest = merge_pdq_fingerprint_chunks(
            context=context, chunks=(_chunk(2, 3), _chunk(0, 2))
        )
        self.assertEqual(manifest["fingerprint_count"], 3)
        bundle = build_pdq_evidence_bundle(
            fingerprint_manifest=manifest,
            policy=PDQSearchPolicy(),
        )
        self.assertEqual(
            bundle["schema_version"], "cvi.public_canine_pdq_evidence_bundle.v1"
        )
        self.assertEqual(len(bundle["evidence"]["search_result"]["candidates"]), 3)
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "pdq-evidence.json"
            publish_pdq_evidence_bundle(path, bundle)
            with self.assertRaises(FileExistsError):
                publish_pdq_evidence_bundle(path, bundle)

    def test_gap_and_chunk_overwrite_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "gap or overlap"):
            merge_pdq_fingerprint_chunks(context=_context(), chunks=(_chunk(1, 3),))
        provenance = {"fixture": True}
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "chunk.json"
            publish_pdq_fingerprint_chunk(path, _chunk(0, 2), tool_provenance=provenance)
            self.assertEqual(read_pdq_fingerprint_chunk(path), _chunk(0, 2))
            with self.assertRaises(FileExistsError):
                publish_pdq_fingerprint_chunk(
                    path, _chunk(0, 2), tool_provenance=provenance
                )


if __name__ == "__main__":
    unittest.main()

"""Construct separated oracle scoring, binding, and ground-truth artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.splits.tracklet_split import SplitManifest
from evaluation.controls.pairing import (
    PairingPolicy,
    construct_verification_pairs,
    dog_attributes_from_payload,
)
from shared.foundation.protected_io import (
    read_strict_json_object as _read_object,
)
from shared.foundation.protected_io import (
    write_private_json_bundle as _write_bundle,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--dog-attributes", required=True, type=Path)
    parser.add_argument("--pairing-policy", required=True, type=Path)
    parser.add_argument("--scoring-output", required=True, type=Path)
    parser.add_argument("--binding-output", required=True, type=Path)
    parser.add_argument("--ground-truth-output", required=True, type=Path)
    parser.add_argument("--summary-output", required=True, type=Path)
    args = parser.parse_args()

    result = construct_verification_pairs(
        SplitManifest.from_dict(_read_object(args.split_manifest)),
        attributes=dog_attributes_from_payload(
            _read_object(args.dog_attributes)
        ),
        policy=PairingPolicy.from_dict(_read_object(args.pairing_policy)),
    )
    _write_bundle(
        (
            (args.scoring_output, result.scoring_payload()),
            (args.binding_output, result.artifact_binding_payload()),
            (args.ground_truth_output, result.ground_truth_payload()),
            (args.summary_output, result.summary_payload()),
        )
    )
    print(
        json.dumps(
            {
                "status": "CREATED",
                "pair_set_sha256": result.result_sha256,
                "pair_count": len(result.scoring_requests),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

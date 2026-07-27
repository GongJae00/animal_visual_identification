"""Assemble role-exposure history from explicit, source-bound declarations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cvi.protected_io import read_strict_json_object, write_private_json_bundle
from cvi.protected_public_split import PublicSplitSourceBundle
from cvi.split_role_exposure import (
    RoleExposureDeclaration,
    create_role_exposure_receipt,
    merge_role_exposure_declarations,
    verify_declaration_source_links,
    verify_split_role_exposure_inputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bundle", required=True, type=Path)
    parser.add_argument(
        "--declaration",
        required=True,
        action="append",
        nargs=2,
        type=Path,
        metavar=("SOURCE_ARTIFACT", "DECLARATION"),
        help=(
            "explicit declaration and the JSON artifact whose content hash it "
            "names; repeat for every known historical artifact"
        ),
    )
    parser.add_argument("--ledger-output", required=True, type=Path)
    parser.add_argument("--receipt-output", required=True, type=Path)
    args = parser.parse_args()

    outputs = (args.ledger_output, args.receipt_output)
    if outputs[0].absolute() == outputs[1].absolute():
        parser.error("ledger and receipt outputs must be distinct")
    existing = [path for path in outputs if path.exists() or path.is_symlink()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite role exposure outputs: "
            + ", ".join(str(path) for path in existing)
        )

    source = PublicSplitSourceBundle.from_dict(
        read_strict_json_object(args.source_bundle)
    )
    declarations: list[RoleExposureDeclaration] = []
    for artifact_path, declaration_path in args.declaration:
        source_artifact = read_strict_json_object(artifact_path)
        declaration = RoleExposureDeclaration.from_dict(
            read_strict_json_object(declaration_path)
        )
        verify_declaration_source_links(declaration, source_artifact)
        declarations.append(declaration)

    ledger = merge_role_exposure_declarations(declarations)
    receipt = create_role_exposure_receipt(ledger)
    verify_split_role_exposure_inputs(source.samples, ledger, receipt)
    write_private_json_bundle(
        (
            (args.ledger_output, ledger.to_dict()),
            (args.receipt_output, receipt.to_dict()),
        )
    )
    print(
        json.dumps(
            {
                "status": "DONE",
                "declaration_count": len(ledger.declarations),
                "record_count": len(ledger.records),
                "ledger_sha256": ledger.ledger_sha256,
                "receipt_sha256": receipt.receipt_sha256,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

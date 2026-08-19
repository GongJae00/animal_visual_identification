"""Exact-profile wrapper for the official PDQ regression byte bundle."""

from __future__ import annotations

from embedding.methods.classical.pdq.source_intake import PdqSourceContract


PDQ_REGRESSION_COMMIT_SHA = "baefb4ed67b6cdc1d4c82dbaef858d50866ac424"
PDQ_REGRESSION_TREE_SHA = "fd19aaa19d4503fe8f5107ae36116fe216d27c24"
PDQ_REGRESSION_SELECTED_PATHS = (
    "LICENSE",
    "pdq/cpp/common/pdqbasetypes.h",
    "pdq/cpp/common/pdqhamming.cpp",
    "pdq/cpp/common/pdqhamming.h",
    "pdq/cpp/common/pdqhashtypes.cpp",
    "pdq/cpp/common/pdqhashtypes.h",
    "pdq/cpp/downscaling/downscaling.cpp",
    "pdq/cpp/downscaling/downscaling.h",
    "pdq/cpp/hashing/pdqhashing.cpp",
    "pdq/cpp/hashing/pdqhashing.h",
    "pdq/cpp/hashing/torben.cpp",
    "pdq/cpp/hashing/torben.h",
    "pdq/cpp/reg_test/expected/out",
    "pdq/data/reg-test-input/dih/bridge-1-original.jpg",
    "pdq/data/reg-test-input/dih/bridge-2-rotate-90.jpg",
    "pdq/data/reg-test-input/dih/bridge-3-rotate-180.jpg",
    "pdq/data/reg-test-input/dih/bridge-4-rotate-270.jpg",
    "pdq/data/reg-test-input/dih/bridge-5-flipx.jpg",
    "pdq/data/reg-test-input/dih/bridge-6-flipy.jpg",
    "pdq/data/reg-test-input/dih/bridge-7-flip-plus-1.jpg",
    "pdq/data/reg-test-input/dih/bridge-8-flip-minus-1.jpg",
)
PDQ_REGRESSION_ASSET_PATHS = PDQ_REGRESSION_SELECTED_PATHS[12:]


def validate_pdq_regression_source_contract(source: PdqSourceContract) -> None:
    """Reject any commit, tree, missing member, or extra member for this profile."""

    if source.commit_sha != PDQ_REGRESSION_COMMIT_SHA:
        raise ValueError("PDQ regression contract commit differs")
    if source.tree_sha != PDQ_REGRESSION_TREE_SHA:
        raise ValueError("PDQ regression contract tree differs")
    selected_paths = tuple(item.relative_path for item in source.selected_members)
    if selected_paths != PDQ_REGRESSION_SELECTED_PATHS:
        missing = sorted(set(PDQ_REGRESSION_SELECTED_PATHS) - set(selected_paths))
        unexpected = sorted(set(selected_paths) - set(PDQ_REGRESSION_SELECTED_PATHS))
        raise ValueError(
            "PDQ regression selected member profile differs; "
            f"missing={missing}; unexpected={unexpected}"
        )

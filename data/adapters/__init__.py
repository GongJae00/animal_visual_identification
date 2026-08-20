"""Dataset layout adapters. One publisher tree → UnifiedCanidSample.

Call ``load(canonical_name, root)``. Do not parse publisher folders in
parsing, identification, or evaluation.
"""

from __future__ import annotations

from pathlib import Path

from data.public_sources.public_canine_manifest import (
    DOGFACE_TEST_MD5,
    DOGFACE_TEST_SHA256,
    DOGFACE_TRAIN_MD5,
    DOGFACE_TRAIN_SHA256,
)
from data.adapters.ap10k_dog import adapt_ap10k_dog
from data.adapters.dogfacenet224 import adapt_dogfacenet224
from data.adapters.dogflw import adapt_dogflw
from data.adapters.io import verified_path as _verified_path
from data.adapters.mpdd import adapt_mpdd
from data.adapters.oxford_pets_dog import adapt_oxford_pets_dog
from data.adapters.petface_dog import (
    PetFaceDogSplitSample,
    adapt_petface_dog,
    load_petface_dog_split,
    read_petface_dog_images,
)
from data.adapters.sibetan import adapt_sibetan
from data.adapters.yt_bb_dog import adapt_yt_bb_dog
from data.types import UnifiedCanidSample

ADAPTERS = {
    "ap10k-dog": adapt_ap10k_dog,
    "dogflw": adapt_dogflw,
    "dogfacenet224": adapt_dogfacenet224,
    "mpdd": adapt_mpdd,
    "oxford-pets-dog": adapt_oxford_pets_dog,
    "sibetan": adapt_sibetan,
    "yt-bb-dog": adapt_yt_bb_dog,
}
RESEARCH_INTAKE_ADAPTERS = {"petface-dog": adapt_petface_dog}

_LAYOUT_MODULES = {
    "ap10k-dog": "data.adapters.ap10k_dog",
    "dogflw": "data.adapters.dogflw",
    "dogfacenet224": "data.adapters.dogfacenet224",
    "mpdd": "data.adapters.mpdd",
    "oxford-pets-dog": "data.adapters.oxford_pets_dog",
    "petface-dog": "data.adapters.petface_dog",
    "sibetan": "data.adapters.sibetan",
    "yt-bb-dog": "data.adapters.yt_bb_dog",
}


def load(canonical_name: str, data_root: Path) -> tuple[UnifiedCanidSample, ...]:
    """Load one dataset by canonical registry name."""

    adapter = ADAPTERS.get(canonical_name)
    if adapter is None:
        adapter = RESEARCH_INTAKE_ADAPTERS.get(canonical_name)
    if adapter is None:
        raise KeyError(f"unknown dataset adapter: {canonical_name!r}")
    return adapter(data_root)


__all__ = [
    "ADAPTERS",
    "DOGFACE_TEST_MD5",
    "DOGFACE_TEST_SHA256",
    "DOGFACE_TRAIN_MD5",
    "DOGFACE_TRAIN_SHA256",
    "RESEARCH_INTAKE_ADAPTERS",
    "PetFaceDogSplitSample",
    "_LAYOUT_MODULES",
    "_verified_path",
    "adapt_ap10k_dog",
    "adapt_dogfacenet224",
    "adapt_dogflw",
    "adapt_mpdd",
    "adapt_oxford_pets_dog",
    "adapt_petface_dog",
    "adapt_sibetan",
    "adapt_yt_bb_dog",
    "load",
    "load_petface_dog_split",
    "read_petface_dog_images",
]

"""Low-level deterministic hashing, timing, and protected I/O primitives."""

from foundation.binomial import required_zero_event_trials, zero_event_exact_upper_bound
from foundation.protected_io import (
    StrictJsonDocument,
    json_document_bytes,
    read_content_hashed_json_bundle,
    read_strict_json_document,
    read_strict_json_object,
    write_private_json_bundle,
    write_private_json_directory_bundle,
)
from foundation.protected_publication import (
    admit_new_external_output,
    fsync_directory,
    rename_directory_noreplace,
)
from foundation.provenance import canonical_json_bytes, content_sha256, git_worktree_provenance
from foundation.retained_file import (
    RetainedFileBinding,
    RetainedFileRead,
    read_retained_regular_file,
    retained_regular_file_binding,
    verify_retained_regular_file_binding,
)
from foundation.timing import TimingSummary

__all__ = [
    "RetainedFileBinding",
    "RetainedFileRead",
    "StrictJsonDocument",
    "TimingSummary",
    "admit_new_external_output",
    "canonical_json_bytes",
    "content_sha256",
    "fsync_directory",
    "git_worktree_provenance",
    "json_document_bytes",
    "read_content_hashed_json_bundle",
    "read_retained_regular_file",
    "read_strict_json_document",
    "read_strict_json_object",
    "rename_directory_noreplace",
    "required_zero_event_trials",
    "retained_regular_file_binding",
    "verify_retained_regular_file_binding",
    "write_private_json_bundle",
    "write_private_json_directory_bundle",
    "zero_event_exact_upper_bound",
]

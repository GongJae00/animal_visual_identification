"""Protected fresh-worker coordinator for embedding cache production."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import Any

from shared.contracts.runtime_library_provenance import (
    RuntimeLibraryManifest,
    RuntimeLibraryPolicy,
)
from data.acquisition import sha256_file
from evaluation.controls.control_scoring import (
    ControlScoringInventory,
    EmbeddingCachePolicy,
    verify_embedding_cache_files,
)
from shared.foundation.protected_io import (
    read_content_hashed_json_bundle,
    read_strict_json_document,
    read_strict_json_object,
)
from shared.foundation.protected_publication import (
    fsync_directory as _fsync_directory,
    rename_directory_noreplace as _rename_directory_noreplace,
)
from shared.foundation.provenance import content_sha256
from shared.foundation.retained_file import (
    retained_regular_file_binding,
    verify_retained_regular_file_binding,
)
from prototype.export.embedding_producer import (
    EmbeddingProducerConfig,
    EmbeddingProductionPolicy,
    EmbeddingProductionReceipt,
    validate_embedding_production_preflight,
)
from operations.workers.process_supervisor import (
    ProcessSupervisorPolicy,
    SupervisedProcessResult,
    SupervisedProcessStatus,
    run_supervised_process,
)
from operations.workers.worker_environment import (
    WorkerEnvironmentIdentity,
    build_sanitized_worker_environment,
)

_PROVENANCE_NAMES = (
    "dependency_lock", "model", "model_lineage", "onnx_config",
    "preprocessing",
)
_CODE_SOURCE_DIRECTORY = Path(__file__).resolve().parents[2]
_CODE_SOURCE_PACKAGE_NAMES = (
    "archive",
    "data",
    "evaluation",
    "identification",
    "enrollment",
    "gallery",
    "parsing",
    "representation",
    "search",
    "shared",
    "operations",
    "prototype",
)
_CODE_SOURCE_NAMES = tuple(
    sorted(
        path.relative_to(_CODE_SOURCE_DIRECTORY).as_posix()
        for package_name in _CODE_SOURCE_PACKAGE_NAMES
        for path in (_CODE_SOURCE_DIRECTORY / package_name).rglob("*.py")
    )
)
LEGACY_EMBEDDING_WORKER_BOOTSTRAP = (
    "import json,os,runpy,sys;"
    "code_root=sys.argv.pop(1);"
    "request_path=sys.argv[2];"
    "request=json.load(open(request_path,encoding='utf-8'));"
    "expected=dict(request['worker_environment_identity']['environment_entries']);"
    "assert dict(os.environ)==expected,"
    "'protected worker initial environment differs from allowlist';"
    "sys.path.insert(0,code_root);"
    "runpy.run_module('operations.embedding_production_worker',"
    "run_name='__main__',alter_sys=True)"
)
MIGRATED_EMBEDDING_WORKER_BOOTSTRAP = (
    "import json,os,runpy,sys;"
    "code_root=sys.argv.pop(1);"
    "request_path=sys.argv[2];"
    "request=json.load(open(request_path,encoding='utf-8'));"
    "expected=dict(request['worker_environment_identity']['environment_entries']);"
    "assert dict(os.environ)==expected,"
    "'protected worker initial environment differs from allowlist';"
    "sys.path.insert(0,code_root);"
    "runpy.run_module('systems.workers.embedding_production_worker',"
    "run_name='__main__',alter_sys=True)"
)
SYSTEMS_EMBEDDING_WORKER_BOOTSTRAP = (
    "import json,os,runpy,sys\n"
    "code_root=sys.argv.pop(1)\n"
    "request_path=sys.argv[2]\n"
    "request=json.load(open(request_path,encoding='utf-8'))\n"
    "expected=dict(request['worker_environment_identity']['environment_entries'])\n"
    "if dict(os.environ)!=expected:\n"
    "    raise RuntimeError('protected worker initial environment differs from allowlist')\n"
    "sys.path.insert(0,code_root)\n"
    "runpy.run_module('systems.workers.embedding_production_worker',"
    "run_name='__main__',alter_sys=True)\n"
)
EMBEDDING_WORKER_BOOTSTRAP = (
    "import json,os,runpy,sys\n"
    "code_root=sys.argv.pop(1)\n"
    "request_path=sys.argv[2]\n"
    "request=json.load(open(request_path,encoding='utf-8'))\n"
    "expected=dict(request['worker_environment_identity']['environment_entries'])\n"
    "if dict(os.environ)!=expected:\n"
    "    raise RuntimeError('protected worker initial environment differs from allowlist')\n"
    "sys.path.insert(0,code_root)\n"
    "runpy.run_module('operations.workers.embedding_production_worker',"
    "run_name='__main__',alter_sys=True)\n"
)
EMBEDDING_WORKER_BOOTSTRAP_SHA256 = content_sha256(
    EMBEDDING_WORKER_BOOTSTRAP
)
LEGACY_EMBEDDING_WORKER_BOOTSTRAP_SHA256 = content_sha256(
    LEGACY_EMBEDDING_WORKER_BOOTSTRAP
)
MIGRATED_EMBEDDING_WORKER_BOOTSTRAP_SHA256 = content_sha256(
    MIGRATED_EMBEDDING_WORKER_BOOTSTRAP
)
SYSTEMS_EMBEDDING_WORKER_BOOTSTRAP_SHA256 = content_sha256(
    SYSTEMS_EMBEDDING_WORKER_BOOTSTRAP
)
_EMBEDDING_WORKER_BOOTSTRAPS = {
    LEGACY_EMBEDDING_WORKER_BOOTSTRAP_SHA256: LEGACY_EMBEDDING_WORKER_BOOTSTRAP,
    MIGRATED_EMBEDDING_WORKER_BOOTSTRAP_SHA256: (
        MIGRATED_EMBEDDING_WORKER_BOOTSTRAP
    ),
    SYSTEMS_EMBEDDING_WORKER_BOOTSTRAP_SHA256: (
        SYSTEMS_EMBEDDING_WORKER_BOOTSTRAP
    ),
    EMBEDDING_WORKER_BOOTSTRAP_SHA256: EMBEDDING_WORKER_BOOTSTRAP,
}


@dataclass(frozen=True, slots=True)
class EmbeddingProductionPrecommitment:
    scoring_inventory_sha256: str
    producer_config_sha256: str
    production_policy_sha256: str
    cache_policy_sha256: str
    backend_identity_sha256: str
    runtime_library_policy_sha256: str
    worker_execution_policy_sha256: str
    worker_environment_identity_sha256: str
    artifact_bindings: tuple[tuple[str, str, int], ...]
    provenance_sha256: tuple[tuple[str, str, int], ...]
    code_source_sha256: tuple[tuple[str, str, int], ...]
    code_source_manifest_sha256: str
    code_source_files: int
    code_source_bytes: int
    worker_bootstrap_sha256: str
    input_bytes_hashed: int
    provenance_bytes_hashed: int
    prior_attempt_ledger_sha256: str
    candidate_attempt_token: str
    precommitment_sequence: int
    selection_blind_to_candidate_outputs: bool = True
    schema_version: str = "cvi.embedding_production_precommitment.v2"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.embedding_production_precommitment.v2":
            raise ValueError("unsupported embedding production precommitment")
        for name in (
            "scoring_inventory_sha256", "producer_config_sha256",
            "production_policy_sha256", "cache_policy_sha256",
            "backend_identity_sha256", "runtime_library_policy_sha256",
            "worker_execution_policy_sha256",
            "worker_environment_identity_sha256",
            "code_source_manifest_sha256", "worker_bootstrap_sha256",
            "prior_attempt_ledger_sha256", "candidate_attempt_token",
        ):
            _sha256(getattr(self, name), name)
        if not self.artifact_bindings:
            raise ValueError("embedding production artifacts are empty")
        tokens: list[str] = []
        for token, digest, byte_size in self.artifact_bindings:
            if not isinstance(token, str) or not token:
                raise ValueError("embedding production artifact token is empty")
            _sha256(digest, "artifact content")
            _positive_int(byte_size, "artifact byte size")
            tokens.append(token)
        if len(tokens) != len(set(tokens)):
            raise ValueError("embedding production artifact tokens repeat")
        if tuple(sorted(self.provenance_sha256)) != self.provenance_sha256 or (
            tuple(name for name, _, _ in self.provenance_sha256)
            != _PROVENANCE_NAMES
        ):
            raise ValueError("embedding production provenance differs")
        for _, digest, byte_size in self.provenance_sha256:
            _sha256(digest, "provenance digest")
            _positive_int(byte_size, "provenance byte size")
        code_source_names = tuple(item[0] for item in self.code_source_sha256)
        if (
            not code_source_names
            or code_source_names != tuple(sorted(set(code_source_names)))
            or any(not _valid_code_source_name(name) for name in code_source_names)
        ):
            raise ValueError("embedding code-source manifest differs")
        for _, digest, byte_size in self.code_source_sha256:
            _sha256(digest, "code source digest")
            _positive_int(byte_size, "code source byte size")
        if self.code_source_manifest_sha256 != content_sha256(
            [list(item) for item in self.code_source_sha256]
        ):
            raise ValueError("embedding code-source aggregate differs")
        if self.worker_bootstrap_sha256 not in _EMBEDDING_WORKER_BOOTSTRAPS:
            raise ValueError("embedding worker bootstrap differs")
        _positive_int(self.code_source_files, "code source files")
        _positive_int(self.code_source_bytes, "code source bytes")
        if self.code_source_files != len(self.code_source_sha256) or (
            self.code_source_bytes != sum(
                item[2] for item in self.code_source_sha256
            )
        ):
            raise ValueError("embedding code-source accounting differs")
        _positive_int(self.input_bytes_hashed, "input bytes hashed")
        _positive_int(self.provenance_bytes_hashed, "provenance bytes hashed")
        _positive_int(self.precommitment_sequence, "precommitment sequence")
        if self.input_bytes_hashed != sum(
            item[2] for item in self.artifact_bindings
        ):
            raise ValueError("embedding input-byte accounting differs")
        if self.provenance_bytes_hashed != sum(
            item[2] for item in self.provenance_sha256
        ):
            raise ValueError("embedding provenance-byte accounting differs")
        if self.selection_blind_to_candidate_outputs is not True:
            raise ValueError("embedding precommitment must be output blind")

    @property
    def precommitment_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        payload = {
            name: getattr(self, name) for name in self.__dataclass_fields__
        }
        payload["artifact_bindings"] = [
            list(item) for item in self.artifact_bindings
        ]
        payload["provenance_sha256"] = [
            list(item) for item in self.provenance_sha256
        ]
        payload["code_source_sha256"] = [
            list(item) for item in self.code_source_sha256
        ]
        return payload

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
    ) -> EmbeddingProductionPrecommitment:
        if set(payload) != set(cls.__dataclass_fields__):
            raise ValueError("embedding production precommitment keys differ")
        if not isinstance(payload["artifact_bindings"], list) or not isinstance(
            payload["provenance_sha256"], list
        ) or not isinstance(
            payload["code_source_sha256"], list
        ):
            raise TypeError("embedding precommitment collections must be lists")
        values = dict(payload)
        values["artifact_bindings"] = tuple(
            tuple(item) for item in values["artifact_bindings"]
        )
        values["provenance_sha256"] = tuple(
            tuple(item) for item in values["provenance_sha256"]
        )
        values["code_source_sha256"] = tuple(
            tuple(item) for item in values["code_source_sha256"]
        )
        return cls(**values)


def _valid_code_source_name(value: object) -> bool:
    if not isinstance(value, str) or not value:
        return False
    path = PurePosixPath(value)
    return (
        not path.is_absolute()
        and value == path.as_posix()
        and len(path.parts) >= 2
        and all(part not in {"", ".", ".."} for part in path.parts)
        and path.suffix == ".py"
    )


@dataclass(frozen=True, slots=True)
class EmbeddingWorkerExecutionPolicy:
    supervisor: ProcessSupervisorPolicy
    maximum_worker_result_bytes: int = 67_108_864
    maximum_snapshot_bytes: int = 68_719_476_736
    maximum_code_snapshot_bytes: int = 16_777_216
    schema_version: str = "cvi.embedding_worker_execution_policy.v2"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.embedding_worker_execution_policy.v2":
            raise ValueError("unsupported embedding worker execution policy")
        _positive_int(
            self.maximum_worker_result_bytes,
            "maximum worker result bytes",
        )
        _positive_int(self.maximum_snapshot_bytes, "maximum snapshot bytes")
        _positive_int(
            self.maximum_code_snapshot_bytes,
            "maximum code snapshot bytes",
        )

    @property
    def policy_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "supervisor": self.supervisor.to_dict(),
            "maximum_worker_result_bytes": self.maximum_worker_result_bytes,
            "maximum_snapshot_bytes": self.maximum_snapshot_bytes,
            "maximum_code_snapshot_bytes": self.maximum_code_snapshot_bytes,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EmbeddingWorkerExecutionPolicy:
        if set(payload) != set(cls.__dataclass_fields__) or not isinstance(
            payload["supervisor"], dict
        ):
            raise ValueError("embedding worker execution policy keys differ")
        values = dict(payload)
        values["supervisor"] = ProcessSupervisorPolicy.from_dict(
            values["supervisor"]
        )
        return cls(**values)


@dataclass(frozen=True, slots=True)
class EmbeddingFreshWorkerReceipt:
    precommitment_sha256: str
    precommitment: EmbeddingProductionPrecommitment
    worker_request_sha256: str
    production_receipt_sha256: str
    production_receipt: EmbeddingProductionReceipt
    runtime_library_manifest_sha256: str
    runtime_library_manifest: RuntimeLibraryManifest
    worker_environment_identity_sha256: str
    worker_environment_identity: WorkerEnvironmentIdentity
    onnxruntime_distribution_name: str
    onnxruntime_distribution_version: str
    actual_providers: tuple[str, ...]
    actual_provider_options_sha256: str
    snapshot_unique_files: int
    snapshot_input_bytes: int
    code_snapshot_files: int
    code_snapshot_bytes: int
    code_snapshot_manifest_sha256: str
    execution_policy_sha256: str
    execution_policy: EmbeddingWorkerExecutionPolicy
    supervised_process_result_sha256: str
    supervised_process_result: SupervisedProcessResult
    publication_status: str
    publication_strategy: str
    completed_attempt_ledger_head_sha256: str
    interpretation: str = (
        "FRESH_WORKER_EMBEDDING_PRODUCTION_NOT_OPTIMIZATION_PROMOTION"
    )
    schema_version: str = "cvi.embedding_fresh_worker_receipt.v2"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.embedding_fresh_worker_receipt.v2":
            raise ValueError("unsupported embedding fresh-worker receipt")
        _validate_outer_common(self)
        if self.runtime_library_manifest.decision != "PASS":
            raise ValueError("embedding runtime manifest did not pass")
        if self.interpretation != (
            "FRESH_WORKER_EMBEDDING_PRODUCTION_NOT_OPTIMIZATION_PROMOTION"
        ):
            raise ValueError("embedding worker interpretation differs")

    @property
    def receipt_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return _outer_to_dict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EmbeddingFreshWorkerReceipt:
        values = _outer_from_dict(cls, payload)
        values["production_receipt"] = EmbeddingProductionReceipt.from_dict(
            values["production_receipt"]
        )
        return cls(**values)


@dataclass(frozen=True, slots=True)
class EmbeddingFreshWorkerDiscovery:
    precommitment_sha256: str
    precommitment: EmbeddingProductionPrecommitment
    worker_request_sha256: str
    production_receipt_sha256: str
    production_receipt: EmbeddingProductionReceipt
    runtime_library_manifest_sha256: str
    runtime_library_manifest: RuntimeLibraryManifest
    worker_environment_identity_sha256: str
    worker_environment_identity: WorkerEnvironmentIdentity
    onnxruntime_distribution_name: str
    onnxruntime_distribution_version: str
    actual_providers: tuple[str, ...]
    actual_provider_options_sha256: str
    snapshot_unique_files: int
    snapshot_input_bytes: int
    code_snapshot_files: int
    code_snapshot_bytes: int
    code_snapshot_manifest_sha256: str
    execution_policy_sha256: str
    execution_policy: EmbeddingWorkerExecutionPolicy
    supervised_process_result_sha256: str
    supervised_process_result: SupervisedProcessResult
    publication_status: str
    publication_strategy: str
    completed_attempt_ledger_head_sha256: str
    interpretation: str = (
        "DISCOVERY_ONLY_NO_CACHE_PUBLICATION_OR_ADMISSION"
    )
    schema_version: str = "cvi.embedding_fresh_worker_discovery.v2"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.embedding_fresh_worker_discovery.v2":
            raise ValueError("unsupported embedding fresh-worker discovery")
        _validate_outer_common(self)
        if self.production_receipt.receipt_sha256 != (
            self.production_receipt_sha256
        ) or self.runtime_library_manifest.decision != "DISCOVERY_ONLY":
            raise ValueError("embedding discovery result differs")
        if self.interpretation != (
            "DISCOVERY_ONLY_NO_CACHE_PUBLICATION_OR_ADMISSION"
        ):
            raise ValueError("embedding discovery interpretation differs")

    @property
    def discovery_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return _outer_to_dict(self)

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
    ) -> EmbeddingFreshWorkerDiscovery:
        values = _outer_from_dict(cls, payload)
        values["production_receipt"] = EmbeddingProductionReceipt.from_dict(
            values["production_receipt"]
        )
        return cls(**values)


def read_embedding_production_outer_bundle(
    path: Path,
    *,
    expected_receipt_sha256: str,
    expected_completed_attempt_ledger_head_sha256: str,
) -> EmbeddingFreshWorkerReceipt:
    """Read one externally anchored, committed production receipt.

    Protected downstream tools must use this boundary instead of accepting the
    inner ``EmbeddingProductionReceipt`` bundle.  The two expected hashes are
    deliberately supplied out of band so a self-consistent rewrite of the JSON
    file cannot become admissible evidence.
    """

    _sha256(expected_receipt_sha256, "expected embedding receipt")
    _sha256(
        expected_completed_attempt_ledger_head_sha256,
        "expected embedding completed-attempt ledger head",
    )
    receipt = EmbeddingFreshWorkerReceipt.from_dict(
        read_content_hashed_json_bundle(
            path,
            schema_version="cvi.embedding_production_bundle.v2",
            payload_field="receipt",
            sha256_field="receipt_sha256",
        )
    )
    if receipt.receipt_sha256 != expected_receipt_sha256:
        raise ValueError("external embedding production receipt anchor differs")
    if receipt.completed_attempt_ledger_head_sha256 != (
        expected_completed_attempt_ledger_head_sha256
    ):
        raise ValueError("external embedding completed-attempt anchor differs")
    return receipt


def build_embedding_production_precommitment(
    *,
    inventory: ControlScoringInventory,
    artifact_paths: Mapping[str, Path],
    producer_config: EmbeddingProducerConfig,
    provenance_paths: Mapping[str, Path],
    production_policy: EmbeddingProductionPolicy,
    cache_policy: EmbeddingCachePolicy,
    runtime_library_policy_sha256: str,
    worker_execution_policy_sha256: str,
    worker_environment_identity_sha256: str,
    prior_attempt_ledger_sha256: str,
    candidate_attempt_token: str,
    precommitment_sequence: int,
) -> EmbeddingProductionPrecommitment:
    bindings = _validate_inventory_artifacts(inventory, artifact_paths)
    provenance = _validate_provenance(provenance_paths, producer_config)
    code_sources = _code_source_bindings()
    validate_embedding_production_preflight(
        inventory=inventory,
        config=producer_config,
        production_policy=production_policy,
        cache_policy=cache_policy,
    )
    for name, path in provenance_paths.items():
        size = path.resolve(strict=True).stat().st_size
        maximum = (
            production_policy.maximum_model_bytes
            if name == "model"
            else production_policy.maximum_provenance_file_bytes
        )
        if size > maximum:
            raise ValueError(f"embedding {name} exceeds production policy")
    return EmbeddingProductionPrecommitment(
        scoring_inventory_sha256=inventory.inventory_sha256,
        producer_config_sha256=producer_config.config_sha256,
        production_policy_sha256=production_policy.policy_sha256,
        cache_policy_sha256=cache_policy.policy_sha256,
        backend_identity_sha256=producer_config.backend.identity_sha256,
        runtime_library_policy_sha256=runtime_library_policy_sha256,
        worker_execution_policy_sha256=worker_execution_policy_sha256,
        worker_environment_identity_sha256=worker_environment_identity_sha256,
        artifact_bindings=bindings,
        provenance_sha256=provenance,
        code_source_sha256=code_sources,
        code_source_manifest_sha256=content_sha256(
            [list(item) for item in code_sources]
        ),
        code_source_files=len(code_sources),
        code_source_bytes=sum(item[2] for item in code_sources),
        worker_bootstrap_sha256=EMBEDDING_WORKER_BOOTSTRAP_SHA256,
        input_bytes_hashed=sum(item[2] for item in bindings),
        provenance_bytes_hashed=sum(
            path.resolve(strict=True).stat().st_size
            for path in provenance_paths.values()
        ),
        prior_attempt_ledger_sha256=prior_attempt_ledger_sha256,
        candidate_attempt_token=candidate_attempt_token,
        precommitment_sequence=precommitment_sequence,
    )


def embedding_artifact_paths_from_dict(
    payload: dict[str, Any],
) -> dict[str, Path]:
    if set(payload) != {"schema_version", "entries"} or payload[
        "schema_version"
    ] != "cvi.embedding_artifact_paths.v1" or not isinstance(
        payload["entries"], list
    ):
        raise ValueError("embedding artifact-path payload differs")
    result: dict[str, Path] = {}
    for entry in payload["entries"]:
        if not isinstance(entry, dict) or set(entry) != {"artifact_token", "path"}:
            raise ValueError("embedding artifact-path entry differs")
        token = entry["artifact_token"]
        path = entry["path"]
        if not isinstance(token, str) or not token or token in result or not isinstance(
            path, str
        ):
            raise ValueError("embedding artifact-path binding differs")
        result[token] = Path(path)
    return result


def run_embedding_production_fresh_worker(
    *,
    backend: str,
    files: Mapping[str, Path],
    precommitment: EmbeddingProductionPrecommitment,
    expected_precommitment_sha256: str,
    python_executable: Path,
    execution_policy: EmbeddingWorkerExecutionPolicy,
    output_directory: Path,
    discovery: bool,
) -> EmbeddingFreshWorkerReceipt | EmbeddingFreshWorkerDiscovery:
    if backend not in {"cpu", "cuda"}:
        raise ValueError("embedding worker backend differs")
    _sha256(expected_precommitment_sha256, "expected precommitment")
    if precommitment.precommitment_sha256 != expected_precommitment_sha256:
        raise ValueError("embedding precommitment differs from external anchor")
    if precommitment.worker_bootstrap_sha256 != EMBEDDING_WORKER_BOOTSTRAP_SHA256:
        raise RuntimeError("historical embedding bootstrap cannot execute")
    _verify_code_source_bindings(precommitment.code_source_sha256)
    required = {
        "inventory", "artifact_paths", "producer_config", "onnx_config",
        "preprocessing", "model", "model_lineage", "dependency_lock",
        "production_policy", "cache_policy", "precommitment",
        "runtime_library_policy",
    }
    if set(files) != required:
        raise ValueError("embedding worker input file names differ")
    bindings = {
        name: retained_regular_file_binding(
            path, subject=f"embedding worker {name}"
        )
        for name, path in sorted(files.items())
    }
    child_environment, environment_identity = build_sanitized_worker_environment(
        os.environ,
        python_executable=python_executable,
    )
    if precommitment.worker_execution_policy_sha256 != (
        execution_policy.policy_sha256
    ):
        raise ValueError("embedding execution policy differs from precommitment")
    if precommitment.worker_environment_identity_sha256 != (
        environment_identity.identity_sha256
    ):
        raise ValueError("embedding worker environment differs from precommitment")
    if precommitment.code_source_bytes > (
        execution_policy.maximum_code_snapshot_bytes
    ):
        raise ValueError("embedding code snapshot exceeds execution policy")
    seen_contents: set[str] = set()
    snapshot_bytes = 0
    for _, digest, byte_size in precommitment.artifact_bindings:
        if digest not in seen_contents:
            seen_contents.add(digest)
            snapshot_bytes += byte_size
    if snapshot_bytes > execution_policy.maximum_snapshot_bytes:
        raise ValueError("embedding snapshot exceeds execution policy")
    runtime_policy = RuntimeLibraryPolicy.from_dict(
        read_strict_json_object(
            Path(bindings["runtime_library_policy"]["path"])
        )
    )
    result_upper_bound = (
        1_048_576
        + sum(len(token.encode("utf-8")) + 512 for token, _, _ in (
            precommitment.artifact_bindings
        ))
        + len(seen_contents) * 1_024
        + runtime_policy.maximum_executable_identities * 4_608
    )
    if result_upper_bound > execution_policy.maximum_worker_result_bytes:
        raise ValueError("embedding worker result estimate exceeds policy")
    output_parent, output_target = _unpublished_output_path(output_directory)
    published = False
    with TemporaryDirectory(
        prefix=".cvi-embedding-worker-",
        dir=output_parent,
    ) as temporary:
        root = Path(temporary)
        os.chmod(root, 0o700)
        scratch = root / "scratch"
        scratch.mkdir(mode=0o700)
        staged_cache = scratch / "cache"
        staged_cache.mkdir(mode=0o700)
        code_root = root / "code"
        code_root.mkdir(mode=0o700)
        code_snapshot_files, code_snapshot_bytes = _snapshot_code_sources(
            precommitment.code_source_sha256,
            code_root,
            maximum_bytes=execution_policy.maximum_code_snapshot_bytes,
        )
        request = {
            "schema_version": "cvi.embedding_fresh_worker_request.v2",
            "backend": backend,
            "files": bindings,
            "expected_precommitment_sha256": expected_precommitment_sha256,
            "worker_environment_identity": environment_identity.to_dict(),
            "worker_environment_identity_sha256": (
                environment_identity.identity_sha256
            ),
            "execution_policy_sha256": execution_policy.policy_sha256,
            "execution_policy": execution_policy.to_dict(),
            "discovery": discovery,
            "scratch_path": str(scratch),
            "cache_path": str(staged_cache),
            "code_snapshot_root": str(code_root),
            "code_snapshot_manifest_sha256": (
                precommitment.code_source_manifest_sha256
            ),
        }
        request_sha256 = content_sha256(request)
        request_path = root / "request.json"
        result_path = root / "result.json"
        request_path.write_text(
            json.dumps(request, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        os.chmod(request_path, 0o600)
        command = (
            environment_identity.python_executable_invocation_path,
            "-I", "-B", "-c", EMBEDDING_WORKER_BOOTSTRAP, str(code_root),
            "--request", str(request_path), "--result", str(result_path),
        )
        supervised = run_supervised_process(
            command,
            policy=execution_policy.supervisor,
            environment=child_environment,
        )
        if supervised.status is not SupervisedProcessStatus.COMPLETED:
            raise RuntimeError(
                "embedding fresh worker failed: "
                f"{supervised.status.value} rc={supervised.return_code}"
            )
        result = read_strict_json_document(
            result_path,
            maximum_bytes=execution_policy.maximum_worker_result_bytes,
        ).payload
        production_receipt, manifest = _validate_worker_result(
            result,
            request_sha256=request_sha256,
            environment_identity=environment_identity,
            backend=backend,
        )
        for name, binding in bindings.items():
            verify_retained_regular_file_binding(
                Path(binding["path"]),
                binding,
                subject=f"embedding worker {name}",
            )
        _verify_code_source_bindings(precommitment.code_source_sha256)
        _verify_code_source_snapshot(
            code_root,
            precommitment.code_source_sha256,
        )
        common = {
            "precommitment_sha256": precommitment.precommitment_sha256,
            "precommitment": precommitment,
            "worker_request_sha256": request_sha256,
            "production_receipt_sha256": production_receipt.receipt_sha256,
            "production_receipt": production_receipt,
            "runtime_library_manifest_sha256": manifest.manifest_sha256,
            "runtime_library_manifest": manifest,
            "worker_environment_identity_sha256": (
                environment_identity.identity_sha256
            ),
            "worker_environment_identity": environment_identity,
            "onnxruntime_distribution_name": result[
                "onnxruntime_distribution_name"
            ],
            "onnxruntime_distribution_version": result[
                "onnxruntime_distribution_version"
            ],
            "actual_providers": tuple(result["actual_providers"]),
            "actual_provider_options_sha256": result[
                "actual_provider_options_sha256"
            ],
            "snapshot_unique_files": result["snapshot_unique_files"],
            "snapshot_input_bytes": result["snapshot_input_bytes"],
            "code_snapshot_files": code_snapshot_files,
            "code_snapshot_bytes": code_snapshot_bytes,
            "code_snapshot_manifest_sha256": (
                precommitment.code_source_manifest_sha256
            ),
            "execution_policy_sha256": execution_policy.policy_sha256,
            "execution_policy": execution_policy,
            "supervised_process_result_sha256": supervised.result_sha256,
            "supervised_process_result": supervised,
        }
        if discovery:
            publication_status = "NOT_PUBLISHED_DISCOVERY"
            publication_strategy = "NONE_DISCOVERY"
            return EmbeddingFreshWorkerDiscovery(
                publication_status=publication_status,
                publication_strategy=publication_strategy,
                completed_attempt_ledger_head_sha256=_completed_attempt_head(
                    precommitment=precommitment,
                    worker_request_sha256=request_sha256,
                    production_receipt_sha256=production_receipt.receipt_sha256,
                    runtime_library_manifest_sha256=manifest.manifest_sha256,
                    supervised_process_result_sha256=supervised.result_sha256,
                    actual_provider_options_sha256=result[
                        "actual_provider_options_sha256"
                    ],
                    snapshot_unique_files=result["snapshot_unique_files"],
                    snapshot_input_bytes=result["snapshot_input_bytes"],
                    code_snapshot_files=code_snapshot_files,
                    code_snapshot_bytes=code_snapshot_bytes,
                    code_snapshot_manifest_sha256=(
                        precommitment.code_source_manifest_sha256
                    ),
                    publication_status=publication_status,
                    publication_strategy=publication_strategy,
                ),
                **common,
            )
        try:
            if output_target.exists() or output_target.is_symlink():
                raise FileExistsError("embedding output appeared during execution")
            publication_strategy = _rename_directory_noreplace(
                staged_cache,
                output_target,
            )
            published = True
            _fsync_directory(output_parent)
            inventory = ControlScoringInventory.from_dict(
                read_strict_json_object(Path(bindings["inventory"]["path"]))
            )
            cache_policy = EmbeddingCachePolicy.from_dict(
                read_strict_json_object(Path(bindings["cache_policy"]["path"]))
            )
            final_verification = verify_embedding_cache_files(
                root=output_target,
                inventory=inventory,
                manifest=production_receipt.cache_manifest,
                policy=cache_policy,
            )
            if final_verification != production_receipt.cache_verification:
                raise RuntimeError("published cache verification differs")
            _fsync_directory(output_target)
            _fsync_directory(output_parent)
            publication_status = "ATOMIC_DIRECTORY_RENAME_COMMITTED"
            return EmbeddingFreshWorkerReceipt(
                publication_status=publication_status,
                publication_strategy=publication_strategy,
                completed_attempt_ledger_head_sha256=_completed_attempt_head(
                    precommitment=precommitment,
                    worker_request_sha256=request_sha256,
                    production_receipt_sha256=production_receipt.receipt_sha256,
                    runtime_library_manifest_sha256=manifest.manifest_sha256,
                    supervised_process_result_sha256=supervised.result_sha256,
                    actual_provider_options_sha256=result[
                        "actual_provider_options_sha256"
                    ],
                    snapshot_unique_files=result["snapshot_unique_files"],
                    snapshot_input_bytes=result["snapshot_input_bytes"],
                    code_snapshot_files=code_snapshot_files,
                    code_snapshot_bytes=code_snapshot_bytes,
                    code_snapshot_manifest_sha256=(
                        precommitment.code_source_manifest_sha256
                    ),
                    publication_status=publication_status,
                    publication_strategy=publication_strategy,
                ),
                **common,
            )
        except BaseException:
            if published:
                _remove_exact_published_cache(output_target, production_receipt)
            raise


def cleanup_published_embedding_cache(
    output_directory: Path,
    receipt: EmbeddingFreshWorkerReceipt,
) -> None:
    """Remove only cache entries named by a receipt after publication failure."""

    root = output_directory.resolve(strict=True)
    _remove_exact_published_cache(root, receipt.production_receipt)


def _completed_attempt_head(
    *,
    precommitment: EmbeddingProductionPrecommitment,
    worker_request_sha256: str,
    production_receipt_sha256: str,
    runtime_library_manifest_sha256: str,
    supervised_process_result_sha256: str,
    actual_provider_options_sha256: str,
    snapshot_unique_files: int,
    snapshot_input_bytes: int,
    code_snapshot_files: int,
    code_snapshot_bytes: int,
    code_snapshot_manifest_sha256: str,
    publication_status: str,
    publication_strategy: str,
) -> str:
    return content_sha256({
        "schema_version": "cvi.embedding_completed_attempt.v2",
        "prior_attempt_ledger_sha256": (
            precommitment.prior_attempt_ledger_sha256
        ),
        "precommitment_sha256": precommitment.precommitment_sha256,
        "candidate_attempt_token": precommitment.candidate_attempt_token,
        "precommitment_sequence": precommitment.precommitment_sequence,
        "worker_request_sha256": worker_request_sha256,
        "production_receipt_sha256": production_receipt_sha256,
        "runtime_library_manifest_sha256": runtime_library_manifest_sha256,
        "supervised_process_result_sha256": (
            supervised_process_result_sha256
        ),
        "actual_provider_options_sha256": actual_provider_options_sha256,
        "snapshot_unique_files": snapshot_unique_files,
        "snapshot_input_bytes": snapshot_input_bytes,
        "code_snapshot_files": code_snapshot_files,
        "code_snapshot_bytes": code_snapshot_bytes,
        "code_snapshot_manifest_sha256": code_snapshot_manifest_sha256,
        "publication_status": publication_status,
        "publication_strategy": publication_strategy,
    })


def _validate_outer_common(value: Any) -> None:
    for name in (
        "precommitment_sha256", "worker_request_sha256",
        "production_receipt_sha256", "runtime_library_manifest_sha256",
        "worker_environment_identity_sha256", "actual_provider_options_sha256",
        "execution_policy_sha256", "supervised_process_result_sha256",
        "code_snapshot_manifest_sha256",
        "completed_attempt_ledger_head_sha256",
    ):
        _sha256(getattr(value, name), name)
    if value.precommitment.precommitment_sha256 != value.precommitment_sha256:
        raise ValueError("embedded production precommitment hash differs")
    if value.production_receipt.receipt_sha256 != (
        value.production_receipt_sha256
    ) or value.production_receipt.scoring_inventory_sha256 != (
        value.precommitment.scoring_inventory_sha256
    ) or value.production_receipt.producer_config_sha256 != (
        value.precommitment.producer_config_sha256
    ) or value.production_receipt.production_policy_sha256 != (
        value.precommitment.production_policy_sha256
    ) or value.production_receipt.cache_policy_sha256 != (
        value.precommitment.cache_policy_sha256
    ):
        raise ValueError("production receipt differs from precommitment")
    if value.runtime_library_manifest.manifest_sha256 != (
        value.runtime_library_manifest_sha256
    ) or value.runtime_library_manifest.policy_sha256 != (
        value.precommitment.runtime_library_policy_sha256
    ):
        raise ValueError("embedding runtime manifest binding differs")
    if value.worker_environment_identity.identity_sha256 != (
        value.worker_environment_identity_sha256
    ) or value.precommitment.worker_environment_identity_sha256 != (
        value.worker_environment_identity_sha256
    ):
        raise ValueError("embedding worker environment binding differs")
    if value.execution_policy.policy_sha256 != value.execution_policy_sha256 or (
        value.precommitment.worker_execution_policy_sha256
        != value.execution_policy_sha256
    ):
        raise ValueError("embedding execution policy binding differs")
    supervised = value.supervised_process_result
    if supervised.result_sha256 != value.supervised_process_result_sha256 or (
        supervised.policy_sha256 != value.execution_policy.supervisor.policy_sha256
    ) or supervised.status is not SupervisedProcessStatus.COMPLETED:
        raise ValueError("embedding supervised process binding differs")
    command = supervised.command
    expected_bootstrap = _EMBEDDING_WORKER_BOOTSTRAPS.get(
        value.precommitment.worker_bootstrap_sha256
    )
    if expected_bootstrap is None or len(command) != 10 or command[:5] != (
        value.worker_environment_identity.python_executable_invocation_path,
        "-I", "-B", "-c", expected_bootstrap,
    ) or not Path(command[5]).is_absolute() or command[6] != (
        "--request"
    ) or command[8] != "--result":
        raise ValueError("embedding worker command differs")
    if supervised.stdout_bytes != 0 or supervised.stderr_bytes != 0:
        raise ValueError("embedding worker emitted unexpected output")
    if supervised.termination_signal_sent or supervised.kill_signal_sent:
        raise ValueError("embedding worker required termination")
    if not value.actual_providers or any(
        not isinstance(item, str) or not item for item in value.actual_providers
    ):
        raise ValueError("embedding actual providers differ")
    _positive_int(value.snapshot_unique_files, "snapshot unique files")
    _positive_int(value.snapshot_input_bytes, "snapshot input bytes")
    _positive_int(value.code_snapshot_files, "code snapshot files")
    _positive_int(value.code_snapshot_bytes, "code snapshot bytes")
    seen_contents: set[str] = set()
    expected_snapshot_bytes = 0
    for _, digest, byte_size in value.precommitment.artifact_bindings:
        if digest not in seen_contents:
            seen_contents.add(digest)
            expected_snapshot_bytes += byte_size
    if value.snapshot_unique_files != len(seen_contents) or (
        value.snapshot_input_bytes != expected_snapshot_bytes
    ):
        raise ValueError("embedding snapshot accounting differs")
    if value.code_snapshot_files != value.precommitment.code_source_files or (
        value.code_snapshot_bytes != value.precommitment.code_source_bytes
    ) or value.code_snapshot_manifest_sha256 != (
        value.precommitment.code_source_manifest_sha256
    ):
        raise ValueError("embedding code snapshot accounting differs")
    expected_distribution = (
        "onnxruntime-gpu"
        if value.actual_providers[0] == "CUDAExecutionProvider"
        else "onnxruntime"
    )
    if value.onnxruntime_distribution_name != expected_distribution or not (
        value.onnxruntime_distribution_version
    ):
        raise ValueError("embedding ONNX Runtime lane differs")
    expected_publication = (
        "NOT_PUBLISHED_DISCOVERY"
        if isinstance(value, EmbeddingFreshWorkerDiscovery)
        else "ATOMIC_DIRECTORY_RENAME_COMMITTED"
    )
    if value.publication_status != expected_publication:
        raise ValueError("embedding publication status differs")
    expected_strategies = (
        {"NONE_DISCOVERY"}
        if isinstance(value, EmbeddingFreshWorkerDiscovery)
        else {
            "RENAMEAT2_NOREPLACE",
            "RESERVED_EMPTY_DIRECTORY_RENAME",
            "PLATFORM_NOREPLACE_RENAME",
        }
    )
    if value.publication_strategy not in expected_strategies:
        raise ValueError("embedding publication strategy differs")
    expected_head = _completed_attempt_head(
        precommitment=value.precommitment,
        worker_request_sha256=value.worker_request_sha256,
        production_receipt_sha256=value.production_receipt_sha256,
        runtime_library_manifest_sha256=value.runtime_library_manifest_sha256,
        supervised_process_result_sha256=value.supervised_process_result_sha256,
        actual_provider_options_sha256=value.actual_provider_options_sha256,
        snapshot_unique_files=value.snapshot_unique_files,
        snapshot_input_bytes=value.snapshot_input_bytes,
        code_snapshot_files=value.code_snapshot_files,
        code_snapshot_bytes=value.code_snapshot_bytes,
        code_snapshot_manifest_sha256=(
            value.code_snapshot_manifest_sha256
        ),
        publication_status=value.publication_status,
        publication_strategy=value.publication_strategy,
    )
    if value.completed_attempt_ledger_head_sha256 != expected_head:
        raise ValueError("embedding completed attempt ledger head differs")


def _outer_to_dict(value: Any) -> dict[str, Any]:
    payload = {
        name: getattr(value, name) for name in value.__dataclass_fields__
    }
    payload["precommitment"] = value.precommitment.to_dict()
    payload["production_receipt"] = value.production_receipt.to_dict()
    payload["runtime_library_manifest"] = (
        value.runtime_library_manifest.to_dict()
    )
    payload["worker_environment_identity"] = (
        value.worker_environment_identity.to_dict()
    )
    payload["actual_providers"] = list(value.actual_providers)
    payload["execution_policy"] = value.execution_policy.to_dict()
    payload["supervised_process_result"] = (
        value.supervised_process_result.to_dict()
    )
    return payload


def _outer_from_dict(cls: type[Any], payload: dict[str, Any]) -> dict[str, Any]:
    if set(payload) != set(cls.__dataclass_fields__):
        raise ValueError("embedding fresh-worker receipt keys differ")
    values = dict(payload)
    values["precommitment"] = EmbeddingProductionPrecommitment.from_dict(
        values["precommitment"]
    )
    values["runtime_library_manifest"] = RuntimeLibraryManifest.from_dict(
        values["runtime_library_manifest"]
    )
    values["worker_environment_identity"] = WorkerEnvironmentIdentity.from_dict(
        values["worker_environment_identity"]
    )
    values["actual_providers"] = tuple(values["actual_providers"])
    values["execution_policy"] = EmbeddingWorkerExecutionPolicy.from_dict(
        values["execution_policy"]
    )
    values["supervised_process_result"] = SupervisedProcessResult.from_dict(
        values["supervised_process_result"]
    )
    return values


def _validate_inventory_artifacts(
    inventory: ControlScoringInventory,
    paths: Mapping[str, Path],
) -> tuple[tuple[str, str, int], ...]:
    if set(paths) != {item.artifact_token for item in inventory.entries}:
        raise ValueError("embedding artifact tokens differ from inventory")
    bindings: list[tuple[str, str, int]] = []
    for item in inventory.entries:
        binding = retained_regular_file_binding(
            paths[item.artifact_token],
            subject=f"embedding worker {item.artifact_token}",
        )
        if binding["byte_size"] != item.byte_size or binding[
            "content_sha256"
        ] != item.content_sha256:
            raise ValueError("embedding artifact content differs from inventory")
        bindings.append((item.artifact_token, item.content_sha256, item.byte_size))
    return tuple(bindings)


def _validate_provenance(
    paths: Mapping[str, Path],
    config: EmbeddingProducerConfig,
) -> tuple[tuple[str, str, int], ...]:
    if set(paths) != set(_PROVENANCE_NAMES):
        raise ValueError("embedding provenance path names differ")
    observed_items = []
    for name in _PROVENANCE_NAMES:
        binding = retained_regular_file_binding(
            paths[name], subject=f"embedding worker {name}"
        )
        observed_items.append(
            (name, binding["content_sha256"], binding["byte_size"])
        )
    observed = tuple(observed_items)
    expected = {
        "model": config.model_sha256,
        "model_lineage": config.model_lineage_sha256,
        "preprocessing": config.preprocessing_sha256,
        "dependency_lock": config.dependency_lock_sha256,
    }
    for name, digest, _ in observed:
        if name in expected and digest != expected[name]:
            raise ValueError(f"embedding {name} provenance differs")
    return observed


def _code_source_bindings() -> tuple[tuple[str, str, int], ...]:
    return _code_source_bindings_at(_CODE_SOURCE_DIRECTORY)


def _code_source_bindings_at(
    root: Path,
) -> tuple[tuple[str, str, int], ...]:
    top_levels = {name.split("/", 1)[0] for name in _CODE_SOURCE_NAMES}
    observed_paths = (
        path
        for top_level in top_levels
        for path in (root / top_level).rglob("*.py")
    )
    observed_names = tuple(sorted(
        path.relative_to(root).as_posix() for path in observed_paths
    ))
    if observed_names != _CODE_SOURCE_NAMES:
        raise RuntimeError("embedding Python source inventory changed")
    result: list[tuple[str, str, int]] = []
    for name in _CODE_SOURCE_NAMES:
        path = root / name
        binding = retained_regular_file_binding(
            path,
            subject=f"embedding worker code source {name}",
        )
        result.append((name, binding["content_sha256"], binding["byte_size"]))
    return tuple(result)


def _verify_code_source_bindings(
    expected: tuple[tuple[str, str, int], ...],
) -> None:
    if _code_source_bindings() != expected:
        raise RuntimeError("embedding protected Python sources changed")


def _snapshot_code_sources(
    expected: tuple[tuple[str, str, int], ...],
    destination_root: Path,
    *,
    maximum_bytes: int,
) -> tuple[int, int]:
    expected_by_name = {
        name: (digest, byte_size) for name, digest, byte_size in expected
    }
    if tuple(expected_by_name) != _CODE_SOURCE_NAMES or any(
        destination_root.iterdir()
    ):
        raise ValueError("embedding code snapshot destination differs")
    total_bytes = sum(item[1] for item in expected_by_name.values())
    if total_bytes > maximum_bytes:
        raise ValueError("embedding code snapshot exceeds execution policy")
    for name in _CODE_SOURCE_NAMES:
        expected_digest, expected_size = expected_by_name[name]
        source = _CODE_SOURCE_DIRECTORY / name
        if source.is_symlink():
            raise ValueError("embedding code source must not be a symlink")
        source_fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            before = os.fstat(source_fd)
            target = destination_root / name
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            target_fd = os.open(
                target,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            digest = hashlib.sha256()
            written = 0
            try:
                while True:
                    chunk = os.read(source_fd, 1_048_576)
                    if not chunk:
                        break
                    digest.update(chunk)
                    view = memoryview(chunk)
                    while view:
                        count = os.write(target_fd, view)
                        view = view[count:]
                    written += len(chunk)
                os.fsync(target_fd)
            finally:
                os.close(target_fd)
            after = os.fstat(source_fd)
        finally:
            os.close(source_fd)
        named = source.stat()
        if _stat_identity(before) != _stat_identity(after) or (
            named.st_dev != after.st_dev or named.st_ino != after.st_ino
        ):
            raise RuntimeError("embedding code source changed during snapshot")
        if written != expected_size or digest.hexdigest() != expected_digest:
            raise RuntimeError("embedding code snapshot content differs")
        os.chmod(target, 0o400)
    _fsync_directory(destination_root)
    _verify_code_source_snapshot(destination_root, expected)
    return len(expected), total_bytes


def _verify_code_source_snapshot(
    root: Path,
    expected: tuple[tuple[str, str, int], ...],
) -> None:
    if _code_source_bindings_at(root) != expected:
        raise RuntimeError("embedding immutable code snapshot changed")


def _validate_worker_result(
    payload: dict[str, Any],
    *,
    request_sha256: str,
    environment_identity: WorkerEnvironmentIdentity,
    backend: str,
) -> tuple[EmbeddingProductionReceipt, RuntimeLibraryManifest]:
    expected = {
        "schema_version", "request_sha256", "backend",
        "worker_environment_identity", "worker_environment_identity_sha256",
        "onnxruntime_distribution_name", "onnxruntime_distribution_version",
        "actual_providers", "actual_provider_options_sha256",
        "snapshot_unique_files", "snapshot_input_bytes",
        "production_receipt", "production_receipt_sha256",
        "runtime_library_manifest", "runtime_library_manifest_sha256",
    }
    if set(payload) != expected or payload["schema_version"] != (
        "cvi.embedding_fresh_worker_result.v2"
    ):
        raise ValueError("embedding worker result schema differs")
    if payload["request_sha256"] != request_sha256 or payload["backend"] != backend:
        raise ValueError("embedding worker result request differs")
    observed_environment = WorkerEnvironmentIdentity.from_dict(
        payload["worker_environment_identity"]
    )
    if observed_environment != environment_identity or (
        observed_environment.identity_sha256
        != payload["worker_environment_identity_sha256"]
    ):
        raise ValueError("embedding worker result environment differs")
    receipt = EmbeddingProductionReceipt.from_dict(payload["production_receipt"])
    manifest = RuntimeLibraryManifest.from_dict(
        payload["runtime_library_manifest"]
    )
    if receipt.receipt_sha256 != payload["production_receipt_sha256"] or (
        manifest.manifest_sha256 != payload["runtime_library_manifest_sha256"]
    ):
        raise ValueError("embedding worker result hash differs")
    if not isinstance(payload["actual_providers"], list):
        raise TypeError("embedding actual providers must be a list")
    _sha256(payload["actual_provider_options_sha256"], "provider options")
    return receipt, manifest


def _unpublished_output_path(path: Path) -> tuple[Path, Path]:
    if not path.name or path.name in {".", ".."}:
        raise ValueError("embedding output path name differs")
    parent = path.parent.resolve(strict=True)
    if not parent.is_dir():
        raise ValueError("embedding output parent must be a directory")
    target = parent / path.name
    if target.exists() or target.is_symlink():
        raise ValueError("embedding output path must not exist")
    return parent, target


def _remove_exact_published_cache(
    root: Path,
    receipt: EmbeddingProductionReceipt,
) -> None:
    expected = {
        entry.relative_path: entry
        for entry in receipt.cache_manifest.entries
    }
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError("published embedding cache ownership differs")
    observed = {path.name: path for path in root.iterdir()}
    if set(observed) != set(expected):
        raise RuntimeError("published embedding cache file set differs")
    for name, path in observed.items():
        entry = expected[name]
        if path.is_symlink() or not path.is_file() or (
            path.stat().st_size != entry.byte_size
        ) or sha256_file(path) != entry.content_sha256:
            raise RuntimeError("published embedding cache content differs")
    for path in observed.values():
        path.unlink()
    root.rmdir()
    _fsync_directory(root.parent)


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev, value.st_ino, value.st_size,
        value.st_mtime_ns, value.st_ctime_ns,
    )


def _positive_int(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _sha256(value: Any, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be lowercase SHA-256")

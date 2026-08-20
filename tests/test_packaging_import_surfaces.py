from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from tests.repo_root import REPO_ROOT

class OptionalImportSurfaceTests(unittest.TestCase):
    def test_evidence_import_does_not_load_optional_runtime_modules(self) -> None:
        script = """
import sys
import representation.evidence as evidence

forbidden = {
    'representation.evidence.calibrator',
    'representation.evidence.oof_simplex',
    'identification.export.appearance',
    'identification.export.landmark',
    'identification.export.backbones.miewid',
    'identification.export.nose.extractor',
    'identification.export.nose.data.dataset',
    'identification.export.nose.signal.frequency',
    'identification.training.nose.losses',
    'identification.export.nose.modeling.model',
    'identification.training.nose.trainer',
    'torch',
    'transformers',
}
loaded = sorted(name for name in forbidden if name in sys.modules)
if loaded:
    raise SystemExit(f'optional modules loaded eagerly: {loaded}')
if evidence.__all__ != [
    'AbstractEvidencer',
    'EvidenceAvailability',
    'EvidenceInsufficiency',
    'EvidenceObservation',
    'EvidenceUnavailableReason',
    'RequiredEvidenceUnavailableError',
]:
    raise SystemExit(f'unexpected representation.evidence exports: {evidence.__all__}')
"""
        environment = os.environ.copy()
        source = str(REPO_ROOT)
        environment["PYTHONPATH"] = os.pathsep.join(
            part for part in (source, environment.get("PYTHONPATH", "")) if part
        )
        subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            env=environment,
        )

    def test_successor_visualization_import_and_validation_do_not_require_torch(
        self,
    ) -> None:
        script = """
import builtins
import sys

original_import = builtins.__import__
blocked_roots = {
    'identification.export',
    'identification.training',
    'archive.full128',
    'torch',
    'torchvision',
    'transformers',
}

def guarded_import(name, *args, **kwargs):
    if any(name == root or name.startswith(root + '.') for root in blocked_roots):
        raise ModuleNotFoundError(f'blocked optional import: {name}')
    return original_import(name, *args, **kwargs)

builtins.__import__ = guarded_import

import visualization.successor_family as successor_family
from shared.foundation.provenance import content_sha256

validator = successor_family.validate_public_successor_evaluation_report
if validator.__module__ != 'evaluation.full128_successor_reporting':
    raise SystemExit(f'unexpected validator module: {validator.__module__}')

aggregates = []
for scope in ('DEV', 'CAL', 'EXPOSED_DIAGNOSTIC'):
    payload = {
        'successor_id': 'B3',
        'scope': scope,
        'status': 'AVAILABLE',
        'reason': None,
        'query_count': 3,
        'identity_count': 3,
        'metrics': {
            'Rank-1': 0.75,
            'Rank-5': 0.75,
            'Rank-10': 0.75,
            'MRR': 0.75,
        },
    }
    aggregates.append({**payload, 'result_sha256': content_sha256(payload)})

candidate = {
    'successor_id': 'B3',
    'cache_descriptor_sha256': '0' * 64,
    'scope_aggregates': aggregates,
    'gallery_bindings': [],
}
selection_payload = {
    'schema_version': 'archive.full128.successor_dev_selection_receipt.v1',
    'selection_scope': 'DEV_ONLY',
    'objective_metric': 'Rank-1',
    'tie_policy': 'SUCCESSOR_ID_ASC',
    'candidates': [{
        'successor_id': 'B3',
        'result_sha256': aggregates[0]['result_sha256'],
        'objective_value': 0.75,
        'denominator': 3,
    }],
    'selected_successor_id': 'B3',
    'calibration_scope_used': False,
    'exposed_scope_used': False,
}
selection = {
    **selection_payload,
    'receipt_sha256': content_sha256(selection_payload),
}
report_payload = {
    'schema_version': 'archive.full128.successor_public_evaluation.v1',
    'visibility': 'PUBLIC_AGGREGATE',
    'source_private_report_sha256': '1' * 64,
    'evaluation_panel_sha256': '2' * 64,
    'candidates': [candidate],
    'dev_selection_receipt': selection,
    'paired_identity_cluster_bootstrap': [],
    'scope_interpretation': {
        'DEV': 'MODEL_SELECTION_ONLY',
        'CAL': 'CALIBRATION_REPORTING;NOT_SELECTION',
        'EXPOSED_DIAGNOSTIC': 'RETROSPECTIVE_EXPOSED;NOT_FINAL_EVALUATION',
    },
    'contains_embeddings': False,
    'contains_sample_or_identity_tokens': False,
    'contains_ranked_qkv_traces': False,
    'limitations': [
        'DEV is used for successor selection; CAL and exposed diagnostics are not selection inputs.',
        'EXPOSED_DIAGNOSTIC is retrospective and is not an independent final evaluation.',
        'The report evaluates exact closed-set cosine retrieval only.',
    ],
}
report = {
    **report_payload,
    'public_report_sha256': content_sha256(report_payload),
}
if validator(report) != report:
    raise SystemExit('public successor report validation changed the report')
if 'archive.full128.evaluation.full128_successors' in sys.modules:
    raise SystemExit('visualization loaded the heavyweight successor evaluator')
loaded = sorted(
    name
    for name in sys.modules
    if any(name == root or name.startswith(root + '.') for root in blocked_roots)
)
if loaded:
    raise SystemExit(f'blocked optional modules loaded: {loaded}')
"""
        environment = os.environ.copy()
        source = str(REPO_ROOT)
        environment["PYTHONPATH"] = os.pathsep.join(
            part for part in (source, environment.get("PYTHONPATH", "")) if part
        )
        subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            env=environment,
        )

class ModelPathTests(unittest.TestCase):
    def test_dogflw_download_is_disabled_without_an_authoritative_hash(self) -> None:
        from shared.contracts.model_paths import DOGFLW_LANDMARK_MD5
        from data.acquisition.models import download_model

        self.assertIsNone(DOGFLW_LANDMARK_MD5)
        with self.assertRaisesRegex(RuntimeError, "DogFLW download is disabled"):
            download_model("dogflw-landmark")

    def test_import_does_not_create_model_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "models"
            environment = os.environ.copy()
            environment["CANINE_IDENTITY_MODELS_DIR"] = str(target)
            source = str(REPO_ROOT)
            environment["PYTHONPATH"] = os.pathsep.join(
                part for part in (source, environment.get("PYTHONPATH", "")) if part
            )
            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import shared.contracts.model_paths; "
                        "from pathlib import Path; "
                        f"assert not Path({str(target)!r}).exists()"
                    ),
                ],
                check=True,
                env=environment,
            )

if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class OptionalImportSurfaceTests(unittest.TestCase):
    def test_evidence_import_does_not_load_optional_runtime_modules(self) -> None:
        script = """
import sys
import cvi.evidence as evidence
import cvi.deployment as deployment

forbidden = {
    'cvi.deployment.cpu',
    'cvi.deployment.cuda',
    'cvi.evidence.appearance',
    'cvi.evidence.landmark_graph',
    'cvi.evidence.miewid',
    'cvi.evidence.nose_print',
    'cvi.nose_id.dataset',
    'cvi.nose_id.frequency',
    'cvi.nose_id.losses',
    'cvi.nose_id.model',
    'cvi.nose_id.trainer',
    'torch',
    'transformers',
}
loaded = sorted(name for name in forbidden if name in sys.modules)
if loaded:
    raise SystemExit(f'optional modules loaded eagerly: {loaded}')
if 'TinyViTBackbone' in evidence.__all__:
    raise SystemExit('disabled nose alias is a supported export')
if deployment.__all__:
    raise SystemExit('disabled deployment constructors are supported exports')
"""
        environment = os.environ.copy()
        source = str(Path(__file__).resolve().parents[1] / "src")
        environment["PYTHONPATH"] = os.pathsep.join(
            part for part in (source, environment.get("PYTHONPATH", "")) if part
        )
        subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            env=environment,
        )

    def test_supported_evidence_symbol_loads_lazily(self) -> None:
        import cvi.evidence as evidence

        self.assertEqual(
            evidence.ReceiptBoundDinov2Small.__module__,
            "cvi.evidence.appearance",
        )


class ModelPathCompatibilityTests(unittest.TestCase):
    def test_compatibility_module_reuses_canonical_paths(self) -> None:
        from cvi import model_paths
        from cvi.utils import model_paths as compatibility

        self.assertIs(compatibility.MODELS_DIR, model_paths.MODELS_DIR)
        self.assertIs(
            compatibility.DOGFLW_LANDMARK_PATH,
            model_paths.DOGFLW_LANDMARK_PATH,
        )

    def test_dogflw_download_is_disabled_without_an_authoritative_hash(self) -> None:
        from cvi.model_paths import DOGFLW_LANDMARK_MD5
        from tools.download_models import download_model

        self.assertIsNone(DOGFLW_LANDMARK_MD5)
        with self.assertRaisesRegex(RuntimeError, "DogFLW download is disabled"):
            download_model("dogflw-landmark")

    def test_compatibility_import_does_not_create_model_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "models"
            environment = os.environ.copy()
            environment["CVI_MODELS_DIR"] = str(target)
            source = str(Path(__file__).resolve().parents[1] / "src")
            environment["PYTHONPATH"] = os.pathsep.join(
                part
                for part in (source, environment.get("PYTHONPATH", ""))
                if part
            )
            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import cvi.utils.model_paths; "
                    "from pathlib import Path; "
                    f"assert not Path({str(target)!r}).exists()",
                ],
                check=True,
                env=environment,
            )


if __name__ == "__main__":
    unittest.main()

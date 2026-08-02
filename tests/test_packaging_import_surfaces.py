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
import evidence_fusion as evidence

forbidden = {
    'identity_methods.appearance',
    'localization.landmark_graph',
    'identity_methods.backbones.miewid',
    'identity_methods.nose.extractor',
    'identity_methods.nose.dataset',
    'identity_methods.nose.frequency',
    'identity_methods.nose.losses',
    'identity_methods.nose.model',
    'identity_methods.nose.trainer',
    'torch',
    'transformers',
}
loaded = sorted(name for name in forbidden if name in sys.modules)
if loaded:
    raise SystemExit(f'optional modules loaded eagerly: {loaded}')
if 'TinyViTBackbone' in evidence.__all__:
    raise SystemExit('disabled nose alias is a supported export')
"""
        environment = os.environ.copy()
        source = str(Path(__file__).resolve().parents[1])
        environment["PYTHONPATH"] = os.pathsep.join(
            part for part in (source, environment.get("PYTHONPATH", "")) if part
        )
        subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            env=environment,
        )

    def test_supported_evidence_symbol_loads_lazily(self) -> None:
        import evidence_fusion as evidence

        self.assertEqual(
            evidence.ReceiptBoundDinov2Small.__module__,
            "identity_methods.appearance",
        )


class ModelPathTests(unittest.TestCase):
    def test_dogflw_download_is_disabled_without_an_authoritative_hash(self) -> None:
        from artifact_contracts.model_paths import DOGFLW_LANDMARK_MD5
        from workflows.download_models import download_model

        self.assertIsNone(DOGFLW_LANDMARK_MD5)
        with self.assertRaisesRegex(RuntimeError, "DogFLW download is disabled"):
            download_model("dogflw-landmark")

    def test_import_does_not_create_model_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "models"
            environment = os.environ.copy()
            environment["CANINE_IDENTITY_MODELS_DIR"] = str(target)
            source = str(Path(__file__).resolve().parents[1])
            environment["PYTHONPATH"] = os.pathsep.join(
                part
                for part in (source, environment.get("PYTHONPATH", ""))
                if part
            )
            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import artifact_contracts.model_paths; "
                    "from pathlib import Path; "
                    f"assert not Path({str(target)!r}).exists()",
                ],
                check=True,
                env=environment,
            )


if __name__ == "__main__":
    unittest.main()

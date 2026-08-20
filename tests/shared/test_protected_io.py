from pathlib import Path

import pytest

from shared.foundation.protected_io import write_private_json_bundle

def test_private_json_publication_rejects_nonfinite_without_output(
    tmp_path: Path,
) -> None:
    path = tmp_path / "report.json"

    with pytest.raises(ValueError, match="JSON compliant"):
        write_private_json_bundle(((path, {"threshold": float("inf")}),))

    assert not path.exists()

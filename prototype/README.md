# Prototype

In: caller-provided crops, receipt-bound channel configs, a local gallery directory.

Out: `IdentityEngine` enroll/search and ONNX Runtime backend helpers.

`runtime/` is the public crop-level closed-set API. `export/` is the device/ONNX
slice used by identification commands and operations workers. Prototype does not
import parsing, training, visualization, or operations.

```python
from prototype.runtime import IdentityEngine, Match
```

Commands: `uv run python -m prototype.commands.export --help`

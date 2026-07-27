from __future__ import annotations

import unittest

import cvi.deployment as deployment
from cvi.deployment import CVIDeploymentCPU, CVIDeploymentCUDA


class DeploymentFacadeTests(unittest.TestCase):
    def test_disabled_constructors_are_not_supported_exports(self) -> None:
        self.assertEqual(deployment.__all__, [])

    def test_cpu_facade_does_not_claim_unused_onnx_inference(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "ignored its ONNX"):
            CVIDeploymentCPU({})

    def test_cuda_facade_does_not_claim_cpu_execution_as_cuda(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "CPU DINO"):
            CVIDeploymentCUDA({})


if __name__ == "__main__":
    unittest.main()

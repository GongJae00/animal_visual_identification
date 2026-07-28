from __future__ import annotations

import unittest

import torch
from torch import nn

from cvi.nose_id.trainer import _backward_cached_output


class _ToyDropoutModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(5, 3)
        self.dropout = nn.Dropout(0.25)

    def forward(self, value: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"embedding": self.dropout(self.linear(value))}


class NoseIDGradientCacheTests(unittest.TestCase):
    def test_cache_recomputation_matches_retained_microbatch_graph(self) -> None:
        inputs = [torch.randn((4, 5)) for _ in range(3)]
        direct = _ToyDropoutModel()
        cached_model = _ToyDropoutModel()
        cached_model.load_state_dict(direct.state_dict())

        torch.manual_seed(31)
        direct_outputs = [direct(value)["embedding"] for value in inputs]
        direct_loss = torch.cat(direct_outputs).square().mean()
        direct_loss.backward()
        expected = direct.linear.weight.grad.detach().clone()

        torch.manual_seed(31)
        leaves = []
        rng_states = []
        for value in inputs:
            rng_states.append(torch.get_rng_state())
            with torch.no_grad():
                output = cached_model(value)["embedding"]
            leaves.append(output.detach().requires_grad_(True))
        torch.cat(leaves).square().mean().backward()
        for value, state, leaf in zip(inputs, rng_states, leaves, strict=True):
            torch.set_rng_state(state)
            output = cached_model(value)
            _backward_cached_output(output, {"embedding": leaf}, ("embedding",))
        torch.testing.assert_close(
            cached_model.linear.weight.grad, expected, rtol=1e-4, atol=1e-5
        )


if __name__ == "__main__":
    unittest.main()

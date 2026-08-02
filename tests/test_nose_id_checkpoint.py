from __future__ import annotations

from pathlib import Path
import random
import tempfile
import unittest
import uuid

import numpy as np
import torch

from identity_methods.nose.checkpoint import (
    SCHEMA_VERSION,
    load_training_checkpoint,
    restore_training_checkpoint,
    save_training_checkpoint,
)


class NoseIDCheckpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = torch.nn.Linear(3, 2)
        self.objective = torch.nn.Linear(2, 2, bias=False)
        self.optimizer = torch.optim.AdamW(
            [*self.model.parameters(), *self.objective.parameters()], lr=1e-3
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=2)
        self.scaler = torch.amp.GradScaler("cpu")
        loss = self.objective(self.model(torch.ones(2, 3))).sum()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        self.identities = {
            str(uuid.uuid5(uuid.NAMESPACE_DNS, "dog-b")): 1,
            str(uuid.uuid5(uuid.NAMESPACE_DNS, "dog-a")): 0,
        }
        self.hashes = {
            "model_sha256": "a" * 64,
            "preprocessor_sha256": "b" * 64,
            "weight_receipt_sha256": "c" * 64,
            "preprocessor_receipt_sha256": "d" * 64,
        }

    def _save(self, path: Path) -> None:
        save_training_checkpoint(
            path,
            model=self.model,
            objective=self.objective,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            identity_to_index=self.identities,
            noseid_config={"embedding_dim": 512},
            train_config={"seed": 7},
            best_dev_n3_map=0.625,
            dino_contract=self.hashes,
            epoch=4,
            global_step=91,
        )

    def test_round_trip_is_weights_only_complete_and_restorable(self) -> None:
        random.seed(17)
        np.random.seed(18)
        torch.manual_seed(19)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "noseid.pt"
            self._save(path)
            raw = torch.load(path, map_location="cpu", weights_only=True)
            loaded = load_training_checkpoint(path)

            self.assertEqual(raw["schema_version"], SCHEMA_VERSION)
            self.assertEqual(set(raw), set(loaded))
            self.assertTrue(
                torch.equal(
                    raw["rng_state"]["torch_cpu"],
                    loaded["rng_state"]["torch_cpu"],
                )
            )
            self.assertEqual(
                loaded["identity_to_index"], dict(sorted(self.identities.items()))
            )
            self.assertEqual(
                loaded["best_metric"], {"name": "DEV_N3_mAP", "value": 0.625}
            )
            self.assertEqual(loaded["dino_contract"], self.hashes)
            self.assertIsInstance(loaded["model_state_dict"], dict)
            self.assertIsInstance(loaded["objective_state_dict"], dict)
            self.assertIsInstance(loaded["optimizer_state_dict"], dict)
            self.assertIsInstance(loaded["scheduler_state_dict"], dict)
            self.assertIsInstance(loaded["scaler_state_dict"], dict)
            self.assertEqual(
                set(loaded["rng_state"]),
                {"python", "numpy", "torch_cpu", "torch_cuda"},
            )

            expected_random = random.random()
            expected_numpy = float(np.random.random())
            expected_torch = float(torch.rand(()))
            with torch.no_grad():
                self.model.weight.zero_()
                self.objective.weight.zero_()
            random.seed(1)
            np.random.seed(1)
            torch.manual_seed(1)
            restore_training_checkpoint(
                loaded,
                model=self.model,
                objective=self.objective,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
            )
            self.assertTrue(
                torch.equal(self.model.weight, loaded["model_state_dict"]["weight"])
            )
            self.assertTrue(
                torch.equal(
                    self.objective.weight,
                    loaded["objective_state_dict"]["weight"],
                )
            )
            self.assertEqual(random.random(), expected_random)
            self.assertEqual(float(np.random.random()), expected_numpy)
            self.assertEqual(float(torch.rand(())), expected_torch)

    def test_save_refuses_overwrite_and_load_rejects_extra_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "noseid.pt"
            self._save(path)
            original = path.read_bytes()
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                self._save(path)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(
                [item.name for item in Path(directory).iterdir()], ["noseid.pt"]
            )

            malformed = torch.load(path, map_location="cpu", weights_only=True)
            malformed["unexpected"] = True
            malformed_path = Path(directory) / "malformed.pt"
            torch.save(malformed, malformed_path)
            with self.assertRaisesRegex(ValueError, "keys differ"):
                load_training_checkpoint(malformed_path)


if __name__ == "__main__":
    unittest.main()

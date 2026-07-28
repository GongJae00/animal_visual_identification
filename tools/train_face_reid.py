"""Train Face ReID on dogfacenet224 — frozen DINOv2 + 256D regional pooling."""

from __future__ import annotations

import json
import time
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from PIL import Image
from cvi.canid_data.adapters import adapt_dogfacenet224
from cvi.face_id.config import FaceIDTrainConfig
from cvi.face_id.dataset import FaceReIDDataset
from cvi.face_id.evaluation import extract_face_embeddings, evaluate_face_retrieval
from cvi.face_id.losses import FaceIDObjective
from cvi.face_id.model import FaceIDModel
from cvi.face_id.sampler import FaceReIDSampler
from cvi.face_id.trainer import build_faceid_optimizer, train_faceid_epoch
from cvi.nose_id.trainer import load_receipt_bound_frozen_dino
from cvi.visualization import contact_sheet


def main() -> None:
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    data_root = Path(
        "/mnt/r/research-data/canine_video_identity_secure/datasets/dogfacenet224"
    )
    model_dir = Path(
        "/mnt/r/research-data/canine_video_identity_secure/checkpoints/"
        "deployment-eligible/hf-facebook-dinov2-small-"
        "ed25f3a31f01632728cabb09d1542f84ab7b0056"
    )
    weight_bundle = Path(
        "/mnt/r/research-data/canine_video_identity_secure/manifests/"
        "pretrained-weights/2026-07-22-v1/dinov2-small-weight-intake.json"
    )
    preproc_bundle = Path(
        "/mnt/r/research-data/canine_video_identity_secure/manifests/"
        "pretrained-weights/2026-07-22-v1/dinov2-small-preprocessor-intake.json"
    )
    output_dir = Path("runs/faceid_v1")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[FaceID] loading dogfacenet224 samples...")
    all_samples = adapt_dogfacenet224(data_root)
    identities = sorted({s.registered_identity_id for s in all_samples if s.registered_identity_id})
    np.random.RandomState(seed).shuffle(identities)
    split = int(len(identities) * 0.8)
    train_ids_full = set(identities[:split])
    from collections import Counter
    id_counts = Counter(s.registered_identity_id for s in all_samples if s.registered_identity_id in train_ids_full)
    train_ids = {uid for uid, count in id_counts.items() if count >= 4}
    dev_ids = set(identities[split:])

    train_samples = tuple(s for s in all_samples if s.registered_identity_id in train_ids)
    dev_samples = tuple(s for s in all_samples if s.registered_identity_id in dev_ids)
    print(f"[FaceID] train={len(train_samples)} images, {len(train_ids)} IDs")
    print(f"[FaceID] dev={len(dev_samples)} images, {len(dev_ids)} IDs")

    train_idx = {uid: i for i, uid in enumerate(sorted(train_ids))}
    dev_idx = {uid: i for i, uid in enumerate(sorted(dev_ids))}

    train_dataset = FaceReIDDataset(data_root, train_samples, train_idx)
    sampler = FaceReIDSampler(
        [s.registered_identity_id for s in train_samples],
        [s.capture_group_id or "unknown" for s in train_samples],
        seed=seed,
    )
    train_loader = DataLoader(
        train_dataset, batch_sampler=sampler, num_workers=0,
        pin_memory=True,
    )

    dev_dataset = FaceReIDDataset(data_root, dev_samples, dev_idx)
    dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False, num_workers=2)

    print("[FaceID] loading DINOv2...")
    backbone, contract = load_receipt_bound_frozen_dino(
        model_directory=model_dir,
        weight_intake_bundle=weight_bundle,
        preprocessor_intake_bundle=preproc_bundle,
    )

    from cvi.face_id.trainer import build_faceid_model
    device = torch.device("cuda")
    model = build_faceid_model(backbone, contract).to(device)
    objective = FaceIDObjective(256, len(train_idx)).to(device)
    optimizer = build_faceid_optimizer(model, objective)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_bf16_supported() is False)

    config = FaceIDTrainConfig(seed=seed)
    best_map = 0.0
    history = []

    for epoch_idx in range(5):
        epoch_num = epoch_idx + 1
        sampler.set_epoch(epoch_idx)

        train_metrics = train_faceid_epoch(
            model, objective, train_loader, optimizer,
            device=device, epoch=epoch_idx, scaler=scaler,
        )

        print(f"[FaceID] epoch {epoch_num}: "
              f"loss={train_metrics['total']:.4f} "
              f"arc={train_metrics['subcenter_arcface']:.4f}")

        if epoch_num % 5 == 0 or epoch_num == 1:
            dev_result = extract_face_embeddings(model, dev_loader, device)
            dev_metrics = evaluate_face_retrieval(
                query_embeddings=dev_result["embeddings"],
                gallery_embeddings=dev_result["embeddings"],
                query_identity_ids=dev_result["identity_ids"],
                gallery_identity_ids=dev_result["identity_ids"],
                query_template_ids=dev_result["template_ids"],
                gallery_template_ids=dev_result["template_ids"],
            )
            dev_map = dev_metrics["mAP"]
            print(f"[FaceID] DEV: Rank-1={dev_metrics['Rank-1']:.4f} mAP={dev_map:.4f}")

            if dev_map > best_map:
                best_map = dev_map
                checkpoint_path = output_dir / "best.pt"
                torch.save({
                    "epoch": epoch_num,
                    "model_state_dict": model.state_dict(),
                    "mAP": best_map,
                }, checkpoint_path)
                print(f"[FaceID] best checkpoint saved (mAP={best_map:.4f})")

        history.append({
            "epoch": epoch_num,
            "train_total": float(train_metrics["total"]),
            "dev_mAP": float(best_map),
        })

    summary = {"seed": seed, "best_mAP": best_map, "epochs": 5, "history": history}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    thumb_samples = list(all_samples)[:64]
    thumb_images = [
        Image.open(data_root / s.image_path).convert("RGB") for s in thumb_samples
    ]
    contact_sheet(thumb_images, stage="face_reid", title="dogfacenet224_faces")
    print("[FaceID] contact sheet saved to visualization/face_reid/")


if __name__ == "__main__":
    main()

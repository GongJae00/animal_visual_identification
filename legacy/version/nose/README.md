# Nose-region variants

Architecture, texture, restoration, and score-fusion trials on same-track
YT-BB-Dog / SiBeTan diagnostics. None beat the frozen Appearance path.

| Setting | Compared | Result | Decision |
|---|---|---|---|
| Same-track K5 | raw vs raw+mask+restoration | Rank-1 +1.24%p. 95% CI −1.86 to +4.97%p includes 0 | No promotion |
| Availability | box crop vs source-coordinate head crop | Explicit Nose/muzzle 6/1,755 → 901/1,755 | Coverage only, not identity |

Training helpers in this folder are research CLIs. Live parsing stays in
`parsing/`. Live appearance stays A0.

```bash
uv run python legacy/version/nose/workflows/audit_nose_observability.py --help
uv run python legacy/version/nose/workflows/evaluate_yt_nose_architecture.py --help
uv run python legacy/version/nose/workflows/evaluate_yt_nose_fusion_scaling.py --help
uv run python legacy/version/nose/workflows/evaluate_yt_nose_restoration.py --help
uv run python legacy/version/nose/workflows/evaluate_yt_nose_texture.py --help
uv run python legacy/version/nose/workflows/extract_yt_native_nose_regions.py --help
uv run python legacy/version/nose/workflows/prepare_nose_annotation_batch.py --help
uv run python legacy/version/nose/workflows/prepare_nose_embedding_views.py --help
uv run python legacy/version/nose/workflows/produce_yt_native_nose_teacher_masks.py --help
uv run python legacy/version/nose/workflows/train_nose_localizer.py --help
uv run python legacy/version/nose/workflows/train_nose_region_consistency.py --help
uv run python legacy/version/nose/workflows/train_nose_region_embedding.py --help
uv run python legacy/version/nose/workflows/train_nose_segmentation_student.py --help
```

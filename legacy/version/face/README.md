# Face F0 / F5

Publisher-test Face identity, train identities held out. Complementary
A+F fusion was not gated.

| Setting | Compared | Result | Decision |
|---|---|---|---|
| Publisher test | F0 vs F5 | Rank-1 37.96% → **40.39%**. ID-bootstrap +2.00 to +2.88%p | F5 research candidate |
| A+F on A/F/N panels | see [afn/](../afn/README.md) | Did not beat A0 | Fusion ungated |

```bash
uv run python legacy/version/face/workflows/build_face_eligibility_overlay.py --help
uv run python legacy/version/face/workflows/build_face_exposure_history.py --help
uv run python legacy/version/face/workflows/build_face_gallery_query_panel.py --help
uv run python legacy/version/face/workflows/build_face_identity_protocol_v2.py --help
uv run python legacy/version/face/workflows/build_face_public_source_binding.py --help
uv run python legacy/version/face/workflows/evaluate_dogface_holdout_fusion.py --help
uv run python legacy/version/face/workflows/evaluate_trained_face_reid.py --help
uv run python legacy/version/face/workflows/train_roi_face_reid.py --help
```

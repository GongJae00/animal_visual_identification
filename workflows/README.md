# Workflows

Functional checkout CLIs for Parsing → Appearance → GenID/ReID. Ablation
sets are in [legacy/version/](../legacy/version/README.md).

```bash
uv run python workflows/<command>.py --help
```

Current diagnostic E2E: `build_animal_parsing_runtime_manifest.py` →
`materialize_full_segment.py` → Appearance artifact →
`evaluate_parsed_body_reid.py`. Product enroll/search is
`runtime.IdentityEngine`.

## Parsing

`build_animal_parsing_runtime_manifest.py`,
`build_canid_unified_manifest.py`,
`materialize_full_segment.py`,
`run_animal_parsing_panel.py`,
`compare_parser_materializations.py`,
`benchmark_animal_parsing_batches.py`,
`benchmark_canid_localizers.py`,
`evaluate_oxford_pet_foreground.py`,
`export_three_region_artifacts.py`.

## Identification / GenID / ReID

`train_embedding_model.py`,
`produce_embedding_cache.py`,
`export_onnx.py`,
`export_pretrained_to_onnx.py`,
`model_parity.py`,
`compare_embedding_caches.py`,
`evaluate_parsed_body_reid.py`,
`evaluate_multichannel.py`,
`migrate_gallery_v3_to_v4.py`.

## Evaluation admission

`prepare_protected_evaluation.py`,
`verify_protected_evaluation.py`,
`construct_verification_pairs.py`,
`create_batch_invariance_precommitment.py`,
`verify_batch_invariance_receipt.py`,
`compare_score_drift.py`,
`evaluate_visual_controls.py`,
`export_oracle_crops.py`.

## Data And Identity

`download_datasets.py`,
`download_models.py`,
`extract_public_dataset_archive.py`,
`build_identity_registry.py`,
`augment_labels_with_registry.py`,
`build_protected_public_split.py`,
`build_unified_full_split.py`,
`build_dataset_stratified_kfold.py`,
`build_localization_kfold.py`,
`assemble_public_split_evidence_graph.py`,
`assemble_role_exposure_ledger.py`,
`adjudicate_public_duplicates.py`,
`audit_public_canine_image_content.py`,
`audit_public_canine_semantics.py`,
`audit_public_canine_phash.py`,
`audit_pretrained_weight.py`,
`audit_pretrained_supporting_asset.py`,
`build_research_cycle_manifest.py`,
`build_research_task_plan.py`.

## Systems

`probe_video.py`,
`benchmark_decode.py`,
`benchmark_onnx_inference.py`,
`compare_onnx_measurements.py`,
`freeze_runtime_library_policy.py`,
`build_native_pdq_worker.py`,
`admit_native_pdq_regression.py`,
`intake_threatexchange_pdq.py`,
`analyze_duplicate_graph_capacity.py`.

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from archive.full128.evaluation import parsing_batch_benchmark as benchmark

def test_prediction_fingerprint_binds_arrays_and_metadata() -> None:
    array = np.ones((2, 3), dtype=np.float32)
    mask = np.ones((2, 3), dtype=np.uint8)
    quality = SimpleNamespace(
        state="USABLE",
        reasons=(),
        flags=(),
        semantic_shape_iou=0.9,
        ownership_retention=1.0,
        foreground_pixels=6,
        component_count=1,
        touches_source_border=False,
    )
    instance = SimpleNamespace(
        instance_index=0,
        query_index=2,
        class_id=17,
        class_name="dog",
        class_score=0.95,
        detector_box_xyxy=(0, 0, 3, 2),
        refinement_box_xyxy=(0, 0, 3, 2),
        mask_box_xyxy=(0, 0, 3, 2),
        quality=quality,
        instance_probability=array,
        foreground_probability=array,
        ownership_probability=array,
        hard_mask=mask,
    )
    prediction = SimpleNamespace(
        source_width=3,
        source_height=2,
        policy_sha256="1" * 64,
        instances=(instance,),
    )

    first = benchmark._exact_prediction_fingerprint(prediction)
    instance.foreground_probability[0, 0] = 0.5
    second = benchmark._exact_prediction_fingerprint(prediction)

    assert first != second
    assert first["instances"][0]["arrays"]["hard_mask"] == second["instances"][0][
        "arrays"
    ]["hard_mask"]

def test_batch_result_requires_repeat_equivalence() -> None:
    repeat = {
        "parser_seconds": 2.0,
        "end_to_end_seconds": 4.0,
        "parser_images_per_second": 256.0,
        "end_to_end_images_per_second": 128.0,
        "peak_cuda_allocated_bytes": 10,
        "peak_cuda_reserved_bytes": 20,
        "exact_prediction_fingerprint_sha256": "1" * 64,
        "semantic_prediction_fingerprint_sha256": "3" * 64,
        "terminal_decision_fingerprint_sha256": "2" * 64,
        "sample_semantic_prediction_sha256s": ["3" * 64],
        "sample_terminal_decision_sha256s": ["4" * 64],
    }
    result = benchmark._batch_result(4, (repeat, repeat, repeat))
    assert result["parser_seconds_median"] == 2.0
    assert result["parser_images_per_second_median"] == 256.0
    assert result["peak_cuda_reserved_bytes_max"] == 20
    assert result["exact_numerical_predictions_repeatable"] is True

def test_comparison_reports_sample_level_mismatches() -> None:
    reference = {
        "sample_semantic_prediction_sha256s": ["1", "2"],
        "sample_terminal_decision_sha256s": ["3", "4"],
    }
    candidate = {
        "sample_semantic_prediction_sha256s": ["1", "x"],
        "sample_terminal_decision_sha256s": ["y", "4"],
    }
    assert benchmark._compare_batch_results(reference, candidate) == {
        "semantic_prediction_mismatch_count": 1,
        "semantic_prediction_mismatch_indices": [1],
        "terminal_decision_mismatch_count": 1,
        "terminal_decision_mismatch_indices": [0],
    }

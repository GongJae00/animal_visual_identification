from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
import subprocess
import sys
import unittest
from collections import Counter
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from evaluation.splits.protected_public_split import (
    EvidenceRelation,
    FrozenPublicSplitEvidenceGraph,
    ProtectedPublicSplitPolicy,
    PublicSplitEvidenceEdge,
    PublicSplitSample,
    PublicSplitSourceBundle,
    _build_protocol_uses,
    _close_components,
    _derive_keys,
    build_protected_public_split,
    create_split_secret,
    read_split_secret,
    seed_commitment,
    validate_protected_split_output_paths,
)
from enrollment.registry.identity_registry import compute_public_subject_token
from evaluation.protected_verification import required_zero_event_trials
from evaluation.splits.split_role_exposure import (
    ExposureDeclarationKind,
    ExposureStage,
    RoleExposureDeclaration,
    RoleExposureDeclarationRecord,
    create_role_exposure_receipt,
    merge_role_exposure_declarations,
)
from evaluation.open_set_calibration import (
    OpenSetCalibrationPolicy,
    authenticate_open_set_calibration_panel,
)

def _token(kind: str, *values: object) -> str:
    payload = "\0".join((kind, *(str(value) for value in values))).encode()
    return hashlib.sha256(payload).hexdigest()

def _bindings() -> tuple[tuple[str, str], ...]:
    names = (
        "exact_duplicate_graph_sha256",
        "geometric_verifier_sha256",
        "image_content_receipts_sha256",
        "pdq_candidates_sha256",
        "phash_candidates_sha256",
        "review_adjudication_sha256",
        "semantic_receipts_sha256",
    )
    return tuple((name, _token("binding", name)) for name in names)

def _sample(
    dataset: str,
    identity: int,
    index: int,
    *,
    split: str | None,
    sequence: int = 0,
    variant: str = "original",
    paired: str | None = None,
    no_mono: bool | None = None,
    region: str = "DOG_CROP",
) -> PublicSplitSample:
    source_id = f"{dataset}:fixture:identity:{identity}:{variant}:sample:{index}:sequence:{sequence}"
    return PublicSplitSample(
        sample_token=_token("sample", source_id),
        identity_token=_token("identity", dataset, identity),
        sequence_token=_token("sequence", dataset, identity, sequence),
        source_sample_id=source_id,
        dataset_identity_id=f"{dataset}:fixture:identity:{identity}",
        dataset_name=dataset,
        source_variant=variant,
        original_split=split,
        raw_frame_index=index,
        paired_source_sample_id=paired,
        in_no_mono_subset=no_mono,
        region=region,
    )

def _fixture(*, add_random_pair: bool = True) -> tuple[PublicSplitSourceBundle, FrozenPublicSplitEvidenceGraph]:
    samples: list[PublicSplitSample] = []
    edges: list[PublicSplitEvidenceEdge] = []
    for identity in range(2000):
        for frame in range(7):
            samples.append(
                _sample("yt-bb-dog", identity, frame, split="train")
            )
    yt_test_first: PublicSplitSample | None = None
    for identity in range(2000, 2723):
        for frame in range(7):
            value = _sample("yt-bb-dog", identity, frame, split="test")
            samples.append(value)
            if identity == 2000 and frame == 6:
                yt_test_first = value
    if add_random_pair:
        assert yt_test_first is not None
        random_value = _sample(
            "yt-bb-dog",
            2000,
            6,
            split="test",
            variant="random_background",
            paired=yt_test_first.source_sample_id,
        )
        samples.append(random_value)
        left, right = sorted((random_value.sample_token, yt_test_first.sample_token))
        edges.append(PublicSplitEvidenceEdge(
            left,
            right,
            EvidenceRelation.DEPENDENCY,
            _token("dependency", left, right),
        ))
    for identity in range(1254):
        samples.append(_sample("dogfacenet224", identity, 0, split="train", region="FACE"))
    for identity in range(1254, 1393):
        for frame in range(4):
            samples.append(_sample("dogfacenet224", identity, frame, split="test", region="FACE"))
    for identity in range(95):
        samples.append(_sample("mpdd", identity, 0, split="train", region="FACE"))
    for identity in range(95, 191):
        for frame in range(5):
            samples.append(_sample("mpdd", identity, frame, split="gallery", region="FACE"))
        samples.append(_sample("mpdd", identity, 9, split="query", region="FACE"))
    for identity in range(39):
        for sequence in (0, 1):
            for frame in range(5):
                samples.append(_sample("sibetan", identity, frame, split=None, sequence=sequence, no_mono=True))
    for identity in range(39, 59):
        samples.append(_sample("sibetan", identity, 0, split=None, no_mono=False))
    source = PublicSplitSourceBundle(_bindings(), tuple(samples))
    graph = FrozenPublicSplitEvidenceGraph(
        _bindings(), tuple(sorted(edges, key=lambda item: (item.left_sample_token, item.right_sample_token)))
    )
    return source, graph

def _build(
    source: PublicSplitSourceBundle,
    graph: FrozenPublicSplitEvidenceGraph,
    *,
    secret: bytes = b"S" * 32,
    historical_stage: ExposureStage = ExposureStage.BYTES_EXPORTED,
    historical_sample: PublicSplitSample | None = None,
):
    sample = historical_sample or source.samples[0]
    declaration = RoleExposureDeclaration(
        source_artifact_sha256=_token(
            "historical-artifact", sample.sample_token, historical_stage.value
        ),
        kind=(
            ExposureDeclarationKind.PRIOR_ASSIGNMENT
            if historical_stage
            in {ExposureStage.BYTES_EXPORTED, ExposureStage.MODEL_TRAINING_USED}
            else ExposureDeclarationKind.PRIOR_EVALUATION
        ),
        revoked=False,
        records=(
            RoleExposureDeclarationRecord(
                sample_token=sample.sample_token,
                identity_token=sample.identity_token,
                public_subject_token=compute_public_subject_token(
                    sample.dataset_identity_id
                ),
                stage=historical_stage,
            ),
        ),
    )
    ledger = merge_role_exposure_declarations((declaration,))
    return build_protected_public_split(
        source=source,
        graph=graph,
        policy=ProtectedPublicSplitPolicy(),
        secret=secret,
        input_file_sha256s=(("evidence_graph_payload_sha256", "1" * 64), ("policy_payload_sha256", "2" * 64), ("source_bundle_payload_sha256", "3" * 64)),
        tool_provenance={"schema_version": "fixture", "code": "unit-test"},
        role_exposure_ledger=ledger,
        role_exposure_receipt=create_role_exposure_receipt(ledger),
    )

class ProtectedPublicSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source, cls.graph = _fixture()

    def test_cli_help_and_fixed_policy_example_round_trip(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-m", "evaluation.commands.evaluate", "protected-split", "--help"],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("--create-secret", completed.stdout)
        path = Path(
            "archive/shared_helpers/configs/contracts/"
            "public_canine_protected_split_policy.example.json"
        )
        policy = ProtectedPublicSplitPolicy.from_dict(json.loads(path.read_text()))
        self.assertEqual(policy, ProtectedPublicSplitPolicy())

    def test_fixed_counts_external_boundary_and_no_label_fields_in_assignment(self) -> None:
        result = _build(self.source, self.graph)
        self.assertEqual(result.status, "PASS_PROTECTED_SPLIT_CONSTRUCTION")
        role_counts = Counter(
            record["identity_role"]
            for record in result.assignment["records"]
            if record["source_variant"] == "original"
        )
        identity_role_counts = Counter()
        seen: set[str] = set()
        for record in result.assignment["records"]:
            if record["identity_token"] not in seen:
                identity_role_counts[record["identity_role"]] += 1
                seen.add(record["identity_token"])
        self.assertEqual(identity_role_counts["YT_FIT"], 1200)
        self.assertEqual(identity_role_counts["YT_DEVELOPMENT"], 200)
        self.assertEqual(identity_role_counts["YT_CALIBRATION_KNOWN"], 300)
        self.assertEqual(identity_role_counts["YT_CALIBRATION_UNKNOWN"], 300)
        self.assertEqual(identity_role_counts["YT_TEST_KNOWN"], 300)
        self.assertEqual(identity_role_counts["YT_TEST_UNKNOWN"], 423)
        self.assertEqual(identity_role_counts["MPDD_EXTERNAL_KNOWN"], 64)
        self.assertEqual(identity_role_counts["MPDD_EXTERNAL_UNKNOWN"], 32)
        serialized = json.dumps(result.assignment, sort_keys=True)
        for forbidden in (
            '"source_sample_id"',
            '"dataset_identity_id"',
            '"sequence_id"',
            '"raw_frame_index"',
            '"original_split"',
            '"score"',
        ):
            self.assertNotIn(forbidden, serialized)
        self.assertIn('"dataset_identity_id"', json.dumps(result.evaluator_binding))
        self.assertGreater(role_counts["YT_TEST_KNOWN"], 300)
        self.assertEqual(
            result.receipt["schema_version"],
            "cvi.protected_public_split_receipt.v3",
        )
        self.assertEqual(
            result.receipt["capacity_mode"],
            "EVIDENCE_CONSTRAINED_MAXIMAL_COVERAGE",
        )
        power = result.receipt["yt_test_unknown_fpir_power"]
        self.assertEqual(power["confidence_level"], 0.95)
        self.assertEqual(power["actual_unknown_identity_trials"], 423)
        self.assertEqual(
            power["targets"],
            [
                {
                    "purpose": "PRIMARY",
                    "target_fpir": 0.01,
                    "required_zero_event_trials": 299,
                    "status": "POWERED",
                },
                {
                    "purpose": "REPORTING",
                    "target_fpir": 0.001,
                    "required_zero_event_trials": 2995,
                    "status": "UNDERPOWERED",
                },
            ],
        )
        self.assertIn("role_exposure_ledger_sha256", result.receipt)
        self.assertIn("role_exposure_receipt_sha256", result.receipt)

    def test_historical_calibration_identity_is_not_regressed_by_role_allocation(self) -> None:
        baseline = _build(self.source, self.graph)
        fit_identity = next(
            record["identity_token"]
            for record in baseline.assignment["records"]
            if record["identity_role"] == "YT_FIT"
        )
        sample = next(
            value
            for value in self.source.samples
            if value.identity_token == fit_identity
        )
        constrained = _build(
            self.source,
            self.graph,
            historical_stage=ExposureStage.CALIBRATION_SCORED,
            historical_sample=sample,
        )
        self.assertEqual(constrained.status, "PASS_PROTECTED_SPLIT_CONSTRUCTION")
        self.assertEqual(
            {
                record["identity_role"]
                for record in constrained.assignment["records"]
                if record["identity_token"] == fit_identity
            },
            {"YT_CALIBRATION_KNOWN"},
        )

    def test_historical_final_test_identity_cannot_return_to_yt_train_lanes(self) -> None:
        sample = next(
            value
            for value in self.source.samples
            if value.dataset_name == "yt-bb-dog"
            and value.original_split == "train"
        )
        result = _build(
            self.source,
            self.graph,
            historical_stage=ExposureStage.FINAL_TEST_SCORED,
            historical_sample=sample,
        )
        self.assertEqual(result.status, "ROLE_EXPOSURE_CAPACITY_FAILED")
        self.assertEqual(result.assignment["records"], [])

    def test_deterministic_order_independent_and_secret_changes_assignment(self) -> None:
        first = _build(self.source, self.graph)
        reversed_source = PublicSplitSourceBundle(
            self.source.evidence_bindings,
            tuple(reversed(self.source.samples)),
        )
        second = _build(reversed_source, self.graph)
        self.assertEqual(first.assignment, second.assignment)
        changed = _build(self.source, self.graph, secret=b"T" * 32)
        first_roles = {
            record["identity_token"]: record["identity_role"]
            for record in first.assignment["records"]
        }
        changed_roles = {
            record["identity_token"]: record["identity_role"]
            for record in changed.assignment["records"]
        }
        self.assertNotEqual(first_roles, changed_roles)
        self.assertNotEqual(first.receipt["seed_commitment"], changed.receipt["seed_commitment"])

    def test_typed_random_background_pair_inherits_role_but_is_control_only(self) -> None:
        result = _build(self.source, self.graph)
        records = {record["sample_token"]: record for record in result.assignment["records"]}
        random_record = next(record for record in records.values() if record["source_variant"] == "random_background")
        original = records[random_record["paired_original_token"]]
        self.assertEqual(random_record["identity_role"], original["identity_role"])
        self.assertEqual(random_record["sample_disposition"], "PAIRED_CONTROL_ONLY")
        self.assertEqual(
            {item["role"] for item in random_record["uses"]},
            {"PAIRED_CONTROL"},
        )

    def test_yt_primary_n300_k3_and_separate_five_shot_diagnostic(self) -> None:
        result = _build(self.source, self.graph)
        labels = {
            row["sample_token"]: row for row in result.evaluator_binding["records"]
        }
        grouped: dict[str, list[dict[str, object]]] = {}
        for record in result.assignment["records"]:
            if record["identity_role"] in {"YT_TEST_KNOWN", "YT_TEST_UNKNOWN"}:
                grouped.setdefault(record["identity_token"], []).append(record)
        self.assertEqual(len(grouped), 723)
        for records in grouped.values():
            galleries = [
                record for record in records
                if any(use["protocol"] == "YT_CLOSED_SET" and use["shot"] == 3 and use["role"] == "GALLERY" for use in record["uses"])
            ]
            queries = [
                record for record in records
                if any(use["protocol"] == "YT_CLOSED_SET" and use["shot"] == 3 and use["role"] == "KNOWN_QUERY" for use in record["uses"])
            ]
            self.assertEqual(len(galleries), 3)
            self.assertEqual(len(queries), 1)
            last_gallery = max(labels[item["sample_token"]]["raw_frame_index"] for item in galleries)
            query_index = labels[queries[0]["sample_token"]]["raw_frame_index"]
            self.assertGreaterEqual(query_index - last_gallery, 2)
        open_set_uses = [
            use
            for record in result.assignment["records"]
            if record["identity_role"] in {"YT_TEST_KNOWN", "YT_TEST_UNKNOWN"}
            for use in record["uses"]
            if use["protocol"] == "YT_OPEN_SET"
        ]
        self.assertEqual({use["shot"] for use in open_set_uses}, {1, 3})
        self.assertEqual({use["gallery_size"] for use in open_set_uses}, {300})
        unknown_events = {
            record["identity_token"]
            for record in result.assignment["records"]
            if record["identity_role"] == "YT_TEST_UNKNOWN"
            and any(
                use["protocol"] == "YT_OPEN_SET"
                and use["shot"] == 3
                and use["role"] == "UNKNOWN_QUERY"
                for use in record["uses"]
            )
        }
        self.assertEqual(len(unknown_events), 423)
        diagnostic = {
            record["identity_token"]
            for record in result.assignment["records"]
            if any(
                use["protocol"] == "YT_CLOSED_SET_DIAGNOSTIC"
                and use["shot"] == 5
                and use["role"] == "KNOWN_QUERY"
                and use["gallery_size"] == 723
                for use in record["uses"]
            )
        }
        self.assertEqual(len(diagnostic), 723)

    def test_test_five_shot_shortfall_is_explicit_diagnostic_subset(self) -> None:
        target = _token("identity", "yt-bb-dog", 2001)
        samples = tuple(
            sample
            for sample in self.source.samples
            if not (
                sample.identity_token == target
                and sample.original_split == "test"
                and sample.source_variant == "original"
                and sample.raw_frame_index > 4
            )
        )
        result = _build(
            PublicSplitSourceBundle(self.source.evidence_bindings, samples),
            self.graph,
        )
        self.assertEqual(result.status, "PASS_PROTECTED_SPLIT_CONSTRUCTION")
        evidence = result.assignment["capacity"]["protocol_evidence_capacity"]
        self.assertEqual(
            evidence["test_diagnostic_five_shot_eligible_identity_count"],
            722,
        )
        target_records = [
            record
            for record in result.assignment["records"]
            if record["identity_token"] == target
        ]
        self.assertTrue(target_records)
        self.assertFalse(any(
            use["protocol"] == "YT_CLOSED_SET_DIAGNOSTIC"
            for record in target_records
            for use in record["uses"]
        ))
        self.assertTrue(any(
            use["protocol"] == "YT_CLOSED_SET" and use["shot"] == 3
            for record in target_records
            for use in record["uses"]
        ))

    def test_test_primary_k3_shortfall_contracts_unknown_without_backfill(self) -> None:
        target = _token("identity", "yt-bb-dog", 2001)
        samples = tuple(
            sample
            for sample in self.source.samples
            if not (
                sample.identity_token == target
                and sample.original_split == "test"
                and sample.source_variant == "original"
                and sample.raw_frame_index > 3
            )
        )
        result = _build(
            PublicSplitSourceBundle(self.source.evidence_bindings, samples),
            self.graph,
        )
        self.assertEqual(result.status, "PASS_PROTECTED_SPLIT_CONSTRUCTION")
        self.assertFalse(any(
            record["identity_token"] == target
            for record in result.assignment["records"]
        ))
        self.assertEqual(
            result.assignment["capacity"]["eligible_yt_test_primary_identities"],
            722,
        )
        self.assertEqual(
            result.assignment["capacity"]["actual_role_counts"]["YT_TEST_KNOWN"],
            300,
        )
        self.assertEqual(
            result.assignment["capacity"]["actual_role_counts"]["YT_TEST_UNKNOWN"],
            422,
        )
        reasons = dict(result.receipt["quarantine"]["reason_counts"])
        self.assertGreater(reasons["PROTOCOL_EVIDENCE_CAPACITY_CONFLICT"], 0)

    def test_development_ab_episodes_are_symmetric_disjoint_and_temporal(self) -> None:
        result = _build(self.source, self.graph)
        labels = {
            row["sample_token"]: row for row in result.evaluator_binding["records"]
        }
        records = [
            record
            for record in result.assignment["records"]
            if record["identity_role"] == "YT_DEVELOPMENT"
        ]
        episode_sets: dict[str, dict[str, set[str]]] = {}
        for episode in ("A_KNOWN_B_UNKNOWN", "B_KNOWN_A_UNKNOWN"):
            roles: dict[str, set[str]] = {
                "GALLERY": set(),
                "KNOWN_QUERY": set(),
                "UNKNOWN_QUERY": set(),
            }
            for record in records:
                for use in record["uses"]:
                    if (
                        use["protocol"] == "YT_DEVELOPMENT_OPEN_SET"
                        and use["episode"] == episode
                        and use["gallery_size"] == 100
                        and use["shot"] == 3
                    ):
                        roles[use["role"]].add(record["identity_token"])
            self.assertEqual(len(roles["GALLERY"]), 100)
            self.assertEqual(roles["GALLERY"], roles["KNOWN_QUERY"])
            self.assertEqual(len(roles["UNKNOWN_QUERY"]), 100)
            self.assertFalse(roles["GALLERY"] & roles["UNKNOWN_QUERY"])
            episode_sets[episode] = roles
        self.assertEqual(
            episode_sets["A_KNOWN_B_UNKNOWN"]["GALLERY"],
            episode_sets["B_KNOWN_A_UNKNOWN"]["UNKNOWN_QUERY"],
        )
        self.assertEqual(
            episode_sets["B_KNOWN_A_UNKNOWN"]["GALLERY"],
            episode_sets["A_KNOWN_B_UNKNOWN"]["UNKNOWN_QUERY"],
        )
        clusters_by_identity: dict[str, set[str]] = {}
        event_tokens: set[str] = set()
        for record in records:
            for use in record["uses"]:
                if (
                    use["protocol"] == "YT_DEVELOPMENT_OPEN_SET"
                    and use["role"] in {"KNOWN_QUERY", "UNKNOWN_QUERY"}
                    and use["shot"] == 3
                ):
                    clusters_by_identity.setdefault(
                        record["identity_token"], set()
                    ).add(use["bootstrap_cluster_token"])
                    event_tokens.add(use["event_token"])
        self.assertEqual(len(clusters_by_identity), 200)
        self.assertTrue(all(len(values) == 1 for values in clusters_by_identity.values()))
        self.assertEqual(len(event_tokens), 400)
        for identity in episode_sets["A_KNOWN_B_UNKNOWN"]["GALLERY"]:
            identity_records = [
                record for record in records if record["identity_token"] == identity
            ]
            galleries = [
                record
                for record in identity_records
                if any(
                    use["protocol"] == "YT_DEVELOPMENT_OPEN_SET"
                    and use["episode"] == "A_KNOWN_B_UNKNOWN"
                    and use["shot"] == 3
                    and use["role"] == "GALLERY"
                    for use in record["uses"]
                )
            ]
            query = next(
                record
                for record in identity_records
                if any(
                    use["protocol"] == "YT_DEVELOPMENT_OPEN_SET"
                    and use["episode"] == "A_KNOWN_B_UNKNOWN"
                    and use["shot"] == 3
                    and use["role"] == "KNOWN_QUERY"
                    for use in record["uses"]
                )
            )
            self.assertEqual(len(galleries), 3)
            self.assertGreaterEqual(
                labels[query["sample_token"]]["raw_frame_index"]
                - max(
                    labels[item["sample_token"]]["raw_frame_index"]
                    for item in galleries
                ),
                2,
            )

    def test_calibration_panels_are_nested_exact_and_unknown_never_gallery(self) -> None:
        result = _build(self.source, self.graph)
        records = [
            record
            for record in result.assignment["records"]
            if record["identity_role"]
            in {"YT_CALIBRATION_KNOWN", "YT_CALIBRATION_UNKNOWN"}
        ]
        observed_sizes: set[int] = set()
        panels: dict[int, set[str]] = {}
        for size in (39, 64, 100, 300):
            gallery: set[str] = set()
            known_query: set[str] = set()
            unknown_query: set[str] = set()
            for record in records:
                for use in record["uses"]:
                    if use["protocol"] != "YT_CALIBRATION_OPEN_SET":
                        continue
                    observed_sizes.add(use["gallery_size"])
                    if use["gallery_size"] == size and use["shot"] == 3:
                        {
                            "GALLERY": gallery,
                            "KNOWN_QUERY": known_query,
                            "UNKNOWN_QUERY": unknown_query,
                        }[use["role"]].add(record["identity_token"])
            self.assertEqual(len(gallery), size)
            self.assertEqual(gallery, known_query)
            self.assertEqual(len(unknown_query), 300)
            self.assertFalse(gallery & unknown_query)
            panels[size] = gallery
        self.assertEqual(observed_sizes, {39, 64, 100, 300})
        self.assertLessEqual(panels[39], panels[64])
        self.assertLessEqual(panels[64], panels[100])
        self.assertLessEqual(panels[100], panels[300])
        unknown_primary_events: dict[str, set[str]] = {}
        for record in records:
            if record["identity_role"] == "YT_CALIBRATION_UNKNOWN":
                self.assertFalse(
                    any(
                        use["protocol"] == "YT_CALIBRATION_OPEN_SET"
                        and use["role"] == "GALLERY"
                        for use in record["uses"]
                    )
                )
                primary_tokens = {
                    use["primary_query_event_token"]
                    for use in record["uses"]
                    if use["protocol"] == "YT_CALIBRATION_OPEN_SET"
                    and use["role"] == "UNKNOWN_QUERY"
                }
                if primary_tokens:
                    self.assertEqual(len(primary_tokens), 1)
                    unknown_primary_events.setdefault(
                        record["identity_token"], set()
                    ).update(primary_tokens)
        self.assertEqual(len(unknown_primary_events), 300)
        self.assertTrue(
            all(len(tokens) == 1 for tokens in unknown_primary_events.values())
        )

    def test_external_open_set_uses_have_exact_nonzero_n_and_k(self) -> None:
        result = _build(self.source, self.graph)
        self.assertEqual(result.status, "PASS_PROTECTED_SPLIT_CONSTRUCTION")
        mpdd_open_shots = {
            use["shot"]
            for record in result.assignment["records"]
            for use in record["uses"]
            if use["protocol"] == "MPDD_OPEN_SET"
        }
        self.assertEqual(mpdd_open_shots, set(ProtectedPublicSplitPolicy().shot_counts))
        self.assertTrue(any(
            use["protocol"] == "MPDD_CLOSED_SET" and use["shot"] == 5
            for record in result.assignment["records"]
            for use in record["uses"]
        ))
        specifications = (
            ("MPDD_OPEN_SET", "MPDD_EXTERNAL_KNOWN", "MPDD_EXTERNAL_UNKNOWN", 64, 32),
            ("SIBETAN_OPEN_SET", "SIBETAN_EXTERNAL_KNOWN", "SIBETAN_EXTERNAL_UNKNOWN", 39, 20),
        )
        for protocol, known_role, unknown_role, n, unknown_count in specifications:
            for shot in (1, 3):
                galleries: dict[str, int] = Counter()
                known_queries: set[str] = set()
                unknown_queries: set[str] = set()
                for record in result.assignment["records"]:
                    for use in record["uses"]:
                        if use["protocol"] != protocol or use["shot"] != shot:
                            continue
                        self.assertEqual(use["gallery_size"], n)
                        if use["role"] == "GALLERY":
                            self.assertEqual(record["identity_role"], known_role)
                            galleries[record["identity_token"]] += 1
                        elif use["role"] == "KNOWN_QUERY":
                            self.assertEqual(record["identity_role"], known_role)
                            known_queries.add(record["identity_token"])
                        elif use["role"] == "UNKNOWN_QUERY":
                            self.assertEqual(record["identity_role"], unknown_role)
                            unknown_queries.add(record["identity_token"])
                self.assertEqual(len(galleries), n)
                self.assertTrue(all(value == shot for value in galleries.values()))
                self.assertEqual(set(galleries), known_queries)
                self.assertEqual(len(unknown_queries), unknown_count)
                self.assertFalse(set(galleries) & unknown_queries)
        capacity = result.assignment["capacity"]["protocol_evidence_capacity"]
        self.assertEqual(
            capacity["external_open_set"]["status"],
            "PASS_EXTERNAL_OPEN_SET_CAPACITY",
        )

    def test_mpdd_known_roles_are_selected_from_k3_eligible_identities(self) -> None:
        samples = tuple(
            sample
            for sample in self.source.samples
            if not (
                sample.dataset_name == "mpdd"
                and sample.dataset_identity_id.endswith((":95", ":96"))
                and sample.original_split == "gallery"
                and sample.raw_frame_index >= 2
            )
        )
        source = PublicSplitSourceBundle(self.source.evidence_bindings, samples)
        result = _build(source, self.graph, secret=b"\0" * 32)
        self.assertEqual(result.status, "PASS_PROTECTED_SPLIT_CONSTRUCTION")
        known = {
            record["identity_token"]
            for record in result.assignment["records"]
            if record["identity_role"] == "MPDD_EXTERNAL_KNOWN"
        }
        self.assertEqual(len(known), 64)
        capacity = result.assignment["capacity"]["protocol_evidence_capacity"]
        self.assertEqual(
            capacity["external_open_set"]["status"],
            "PASS_EXTERNAL_OPEN_SET_CAPACITY",
        )

    def test_authenticated_calibration_panel_is_exactly_derived_from_assignment(self) -> None:
        result = _build(self.source, self.graph)
        panel = authenticate_open_set_calibration_panel(
            result.assignment,
            split_policy=ProtectedPublicSplitPolicy(),
            calibration_policy=OpenSetCalibrationPolicy(),
            gallery_size=300,
            shot=3,
        )
        self.assertEqual(len(panel.gallery_identity_slot_tokens), 300)
        self.assertEqual(len(panel.unknown_query_event_tokens), 300)
        self.assertEqual(panel.split_assignment_sha256, result.receipt["assignment_sha256"])

        tampered = copy.deepcopy(result.assignment)
        tampered["policy_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "policy hash"):
            authenticate_open_set_calibration_panel(
                tampered,
                split_policy=ProtectedPublicSplitPolicy(),
                calibration_policy=OpenSetCalibrationPolicy(),
                gallery_size=300,
                shot=3,
            )

        tampered = copy.deepcopy(result.assignment)
        changed = False
        for record in tampered["records"]:
            if record["identity_role"] != "YT_CALIBRATION_UNKNOWN":
                continue
            for use in record["uses"]:
                if (
                    use["protocol"] == "YT_CALIBRATION_OPEN_SET"
                    and use["gallery_size"] == 300
                    and use["shot"] == 3
                ):
                    use["role"] = "GALLERY"
                    use["primary_query_event_token"] = None
                    use["bootstrap_cluster_token"] = None
                    changed = True
                    break
            if changed:
                break
        self.assertTrue(changed)
        with self.assertRaisesRegex(ValueError, "unknown identity entered gallery"):
            authenticate_open_set_calibration_panel(
                tampered,
                split_policy=ProtectedPublicSplitPolicy(),
                calibration_policy=OpenSetCalibrationPolicy(),
                gallery_size=300,
                shot=3,
            )

    def test_train_primary_evidence_shortfall_empties_every_assignment(self) -> None:
        samples = tuple(
            sample
            for sample in self.source.samples
            if not (
                sample.dataset_name == "yt-bb-dog"
                and sample.original_split == "train"
                and sample.raw_frame_index > 0
            )
        )
        source = PublicSplitSourceBundle(self.source.evidence_bindings, samples)
        result = _build(source, self.graph)
        self.assertEqual(result.status, "SPLIT_CAPACITY_FAILED")
        self.assertEqual(result.assignment["records"], [])
        self.assertEqual(result.evaluator_binding["records"], [])
        self.assertEqual(
            result.assignment["capacity"]["eligible_yt_train_identities"], 0
        )
        self.assertNotIn(
            "protocol_evidence_capacity", result.assignment["capacity"]
        )

    def test_calibration_shortfall_has_a_distinct_fail_closed_status(self) -> None:
        baseline = _build(self.source, self.graph)
        roles = {
            record["identity_token"]: record["identity_role"]
            for record in baseline.assignment["records"]
        }
        target = next(
            identity
            for identity, role in roles.items()
            if role == "YT_CALIBRATION_KNOWN"
        )
        samples = tuple(
            sample
            for sample in self.source.samples
            if not (
                sample.identity_token == target and sample.raw_frame_index > 0
            )
        )
        components, quarantined, _ = _close_components(
            samples, self.graph.edges
        )
        component_by_sample = {
            sample.sample_token: component
            for component in components
            for sample in component.samples
        }
        _, capacity = _build_protocol_uses(
            samples,
            component_by_sample,
            quarantined,
            roles,
            ProtectedPublicSplitPolicy(),
            _derive_keys(b"S" * 32, "0" * 64),
        )
        self.assertEqual(capacity["status"], "CALIBRATION_CAPACITY_FAILED")
        self.assertEqual(
            capacity["calibration_known_eligible_identity_count"], 299
        )

    def test_unresolved_review_quarantines_whole_identities_before_roles(self) -> None:
        left = next(sample for sample in self.source.samples if sample.dataset_name == "yt-bb-dog" and sample.original_split == "train")
        right = next(sample for sample in self.source.samples if sample.dataset_name == "yt-bb-dog" and sample.original_split == "train" and sample.identity_token != left.identity_token)
        edge = PublicSplitEvidenceEdge(
            *sorted((left.sample_token, right.sample_token)),
            EvidenceRelation.REVIEW_UNRESOLVED,
            _token("review", left.sample_token, right.sample_token),
        )
        graph = FrozenPublicSplitEvidenceGraph(self.graph.evidence_bindings, tuple(sorted((*self.graph.edges, edge), key=lambda item: (item.left_sample_token, item.right_sample_token))))
        result = _build(self.source, graph)
        self.assertEqual(result.status, "PASS_PROTECTED_SPLIT_CONSTRUCTION")
        self.assertEqual(result.assignment["capacity"]["actual_fit_identities"], 1198)
        assigned = {record["identity_token"] for record in result.assignment["records"]}
        self.assertNotIn(left.identity_token, assigned)
        self.assertNotIn(right.identity_token, assigned)

    def test_capacity_fails_below_1800_and_emits_no_split_records(self) -> None:
        yt_train = [
            sample for sample in self.source.samples
            if sample.dataset_name == "yt-bb-dog" and sample.original_split == "train"
            and sample.raw_frame_index == 0
        ]
        new_edges = list(self.graph.edges)
        for index in range(0, 402, 2):
            left, right = sorted((yt_train[index].sample_token, yt_train[index + 1].sample_token))
            new_edges.append(PublicSplitEvidenceEdge(left, right, EvidenceRelation.REVIEW_UNRESOLVED, _token("unresolved", left, right)))
        graph = FrozenPublicSplitEvidenceGraph(self.graph.evidence_bindings, tuple(sorted(new_edges, key=lambda item: (item.left_sample_token, item.right_sample_token))))
        result = _build(self.source, graph)
        self.assertEqual(result.status, "SPLIT_CAPACITY_FAILED")
        self.assertEqual(result.assignment["records"], [])
        self.assertEqual(result.evaluator_binding["records"], [])
        self.assertEqual(result.assignment["capacity"]["eligible_yt_train_identities"], 1598)

    def test_missing_required_dependency_and_graph_binding_mismatch_fail_closed(self) -> None:
        graph_without_dependency = FrozenPublicSplitEvidenceGraph(self.graph.evidence_bindings, ())
        with self.assertRaisesRegex(ValueError, "dependency edge is missing"):
            _build(self.source, graph_without_dependency)
        changed = list(self.graph.evidence_bindings)
        changed[0] = (changed[0][0], "f" * 64)
        mismatched = FrozenPublicSplitEvidenceGraph(tuple(changed), self.graph.edges)
        with self.assertRaisesRegex(ValueError, "bindings differ"):
            _build(self.source, mismatched)

    def test_mpdd_query_gallery_dependency_stays_in_one_role_without_leakage(self) -> None:
        mpdd = [
            sample
            for sample in self.source.samples
            if sample.dataset_name == "mpdd"
            and sample.dataset_identity_id.endswith(":95")
        ]
        gallery = next(sample for sample in mpdd if sample.original_split == "gallery")
        query = next(sample for sample in mpdd if sample.original_split == "query")
        left, right = sorted((gallery.sample_token, query.sample_token))
        edge = PublicSplitEvidenceEdge(
            left,
            right,
            EvidenceRelation.EXACT_CONFIRMED,
            _token("exact", left, right),
        )
        graph = FrozenPublicSplitEvidenceGraph(
            self.graph.evidence_bindings,
            tuple(sorted((*self.graph.edges, edge), key=lambda item: (
                item.left_sample_token,
                item.right_sample_token,
                item.relation.value,
                item.evidence_token,
            ))),
        )
        result = _build(self.source, graph)
        self.assertEqual(result.status, "PASS_PROTECTED_SPLIT_CONSTRUCTION")
        records = {
            record["sample_token"]: record
            for record in result.assignment["records"]
        }
        self.assertEqual(
            records[gallery.sample_token]["identity_role"],
            records[query.sample_token]["identity_role"],
        )
        query_component = records[query.sample_token]["component_token"]
        query_uses = [
            use
            for record in result.assignment["records"]
            if record["component_token"] == query_component
            for use in record["uses"]
        ]
        self.assertFalse(any(use["role"] == "GALLERY" for use in query_uses))

    def test_source_variant_whitelist_is_closed(self) -> None:
        samples = list(self.source.samples)
        samples[0] = replace(samples[0], source_variant="generated_enhancement")
        with self.assertRaisesRegex(ValueError, "unsupported public dataset or source variant"):
            PublicSplitSourceBundle(self.source.evidence_bindings, tuple(samples))

    def test_identity_label_cannot_be_split_across_two_opaque_tokens(self) -> None:
        samples = list(self.source.samples)
        target = samples[7]
        samples[7] = replace(
            target,
            dataset_identity_id=samples[0].dataset_identity_id,
        )
        with self.assertRaisesRegex(
            ValueError, "one identity label maps to multiple opaque tokens"
        ):
            PublicSplitSourceBundle(self.source.evidence_bindings, tuple(samples))

    def test_one_pair_cannot_have_conflicting_duplicate_adjudications(self) -> None:
        left, right = sorted(
            (self.source.samples[0].sample_token, self.source.samples[7].sample_token)
        )
        edges = tuple(
            sorted(
                (
                    PublicSplitEvidenceEdge(
                        left,
                        right,
                        EvidenceRelation.REVIEW_CONFIRMED,
                        _token("confirmed", left, right),
                    ),
                    PublicSplitEvidenceEdge(
                        left,
                        right,
                        EvidenceRelation.REVIEW_REJECTED,
                        _token("rejected", left, right),
                    ),
                ),
                key=lambda item: (
                    item.left_sample_token,
                    item.right_sample_token,
                    item.relation.value,
                    item.evidence_token,
                ),
            )
        )
        with self.assertRaisesRegex(ValueError, "contradictory"):
            FrozenPublicSplitEvidenceGraph(self.graph.evidence_bindings, edges)

    def test_cross_identity_component_is_indivisible_within_official_lane(self) -> None:
        with self.assertRaises(ValueError):
            EvidenceRelation("GEOMETRIC_REJECTED")
        left_sample = self.source.samples[0]
        right_sample = self.source.samples[7]
        left, right = sorted(
            (left_sample.sample_token, right_sample.sample_token)
        )
        edge = PublicSplitEvidenceEdge(
            left,
            right,
            EvidenceRelation.DEPENDENCY,
            _token("geometric", left, right),
        )
        graph = FrozenPublicSplitEvidenceGraph(
            self.graph.evidence_bindings,
            tuple(
                sorted(
                    (*self.graph.edges, edge),
                    key=lambda item: (
                        item.left_sample_token,
                        item.right_sample_token,
                        item.relation.value,
                        item.evidence_token,
                    ),
                )
            ),
        )
        result = _build(
            self.source,
            graph,
            historical_stage=ExposureStage.CALIBRATION_SCORED,
            historical_sample=left_sample,
        )
        self.assertEqual(result.status, "PASS_PROTECTED_SPLIT_CONSTRUCTION")
        roles = {
            record["identity_token"]: record["identity_role"]
            for record in result.assignment["records"]
            if record["identity_token"]
            in {left_sample.identity_token, right_sample.identity_token}
        }
        self.assertEqual(len(roles), 2)
        self.assertEqual(set(roles.values()), {"YT_CALIBRATION_KNOWN"})
        components = {
            record["component_token"]
            for record in result.assignment["records"]
            if record["sample_token"] in {left_sample.sample_token, right_sample.sample_token}
        }
        self.assertEqual(len(components), 1)

    def test_cross_official_lane_block_is_closed_and_quarantined(self) -> None:
        left_sample = next(
            sample
            for sample in self.source.samples
            if sample.dataset_name == "yt-bb-dog"
            and sample.original_split == "train"
            and sample.source_variant == "original"
        )
        right_sample = next(
            sample
            for sample in self.source.samples
            if sample.dataset_name == "yt-bb-dog"
            and sample.original_split == "test"
            and sample.source_variant == "original"
        )
        left, right = sorted((left_sample.sample_token, right_sample.sample_token))
        edge = PublicSplitEvidenceEdge(
            left,
            right,
            EvidenceRelation.DEPENDENCY,
            _token("conservative-dependency", left, right),
        )
        graph = FrozenPublicSplitEvidenceGraph(
            self.graph.evidence_bindings,
            tuple(sorted((*self.graph.edges, edge), key=lambda item: (
                item.left_sample_token,
                item.right_sample_token,
                item.relation.value,
                item.evidence_token,
            ))),
        )
        result = _build(self.source, graph)
        self.assertEqual(result.status, "PASS_PROTECTED_SPLIT_CONSTRUCTION")
        quarantine = result.receipt["quarantine"]
        self.assertEqual(quarantine["identity_count"], 2)
        self.assertEqual(
            quarantine["sample_count"],
            sum(
                sample.identity_token
                in {left_sample.identity_token, right_sample.identity_token}
                for sample in self.source.samples
            ),
        )
        self.assertEqual(quarantine["allocation_block_count"], 1)
        reasons = dict(quarantine["reason_counts"])
        self.assertGreater(reasons["OFFICIAL_LANE_CONFLICT"], 0)
        capacity = result.assignment["capacity"]
        self.assertEqual(capacity["actual_role_counts"]["YT_FIT"], 1199)
        self.assertEqual(capacity["actual_role_counts"]["YT_TEST_KNOWN"], 300)
        self.assertEqual(capacity["actual_role_counts"]["YT_TEST_UNKNOWN"], 422)
        self.assertEqual(capacity["contracted_role_counts"]["YT_TEST_UNKNOWN"], 1)

    def test_dogface_maximal_coverage_contracts_only_fit_and_test(self) -> None:
        train_sample = next(
            sample
            for sample in self.source.samples
            if sample.dataset_name == "dogfacenet224"
            and sample.original_split == "train"
        )
        test_sample = next(
            sample
            for sample in self.source.samples
            if sample.dataset_name == "dogfacenet224"
            and sample.original_split == "test"
        )
        left, right = sorted((train_sample.sample_token, test_sample.sample_token))
        edge = PublicSplitEvidenceEdge(
            left,
            right,
            EvidenceRelation.DEPENDENCY,
            _token("dogface-lane-dependency", left, right),
        )
        graph = FrozenPublicSplitEvidenceGraph(
            self.graph.evidence_bindings,
            tuple(sorted((*self.graph.edges, edge), key=lambda item: (
                item.left_sample_token,
                item.right_sample_token,
                item.relation.value,
                item.evidence_token,
            ))),
        )
        result = _build(self.source, graph)
        self.assertEqual(result.status, "PASS_PROTECTED_SPLIT_CONSTRUCTION")
        actual = result.assignment["capacity"]["actual_role_counts"]
        self.assertEqual(actual["DOGFACE_FIT"], 1003)
        self.assertEqual(actual["DOGFACE_DEVELOPMENT"], 125)
        self.assertEqual(actual["DOGFACE_CALIBRATION"], 125)
        self.assertEqual(actual["DOGFACE_TEST"], 138)

    def test_hmac_implementation_has_no_prng_or_python_hash_dependency(self) -> None:
        source = Path("evaluation/splits/protected_public_split.py").read_text()
        tree = ast.parse(source)
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        self.assertNotIn("random", imported)
        calls = [node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
        self.assertNotIn("hash", calls)
        self.assertIn("hmac.new", source)

    def test_secret_is_32_bytes_mode_0600_no_replace_and_symlink_rejected(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            secret_path = root / "split.seed"
            secret = create_split_secret(secret_path)
            self.assertEqual(len(secret), 32)
            self.assertEqual(read_split_secret(secret_path), secret)
            self.assertEqual(os.stat(secret_path).st_mode & 0o777, 0o600)
            self.assertEqual(seed_commitment(secret), seed_commitment(secret))
            with self.assertRaises(FileExistsError):
                create_split_secret(secret_path)
            public = root / "public.seed"
            public.write_bytes(b"X" * 32)
            os.chmod(public, 0o644)
            with self.assertRaises(PermissionError):
                read_split_secret(public)
            link = root / "link.seed"
            link.symlink_to(secret_path)
            with self.assertRaises(OSError):
                read_split_secret(link)

    def test_output_preflight_rejects_overwrite_alias_and_mixed_parents(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            other = root / "other"
            other.mkdir()
            outputs = (root / "assignment.json", root / "labels.json", root / "receipt.json")
            validate_protected_split_output_paths(outputs)
            outputs[0].write_text("protected", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                validate_protected_split_output_paths(outputs)
            outputs[0].unlink()
            with self.assertRaisesRegex(ValueError, "distinct"):
                validate_protected_split_output_paths((outputs[0], outputs[0], outputs[2]))
            with self.assertRaisesRegex(ValueError, "share one directory"):
                validate_protected_split_output_paths((outputs[0], outputs[1], other / "receipt.json"))
            link = root / "link.json"
            link.symlink_to(root / "absent.json")
            with self.assertRaises(FileExistsError):
                validate_protected_split_output_paths((link, outputs[1], outputs[2]))

    def test_schema_rejects_score_fields_and_policy_drift(self) -> None:
        payload = self.source.samples[0].to_dict()
        payload["score"] = 0.9
        with self.assertRaisesRegex(ValueError, "fields differ"):
            PublicSplitSample.from_dict(payload)
        policy = ProtectedPublicSplitPolicy().to_dict()
        policy["yt_test_known_identities"] = 301
        policy["yt_test_unknown_identities"] = 422
        policy["yt_primary_open_set_gallery_size"] = 301
        with self.assertRaisesRegex(ValueError, "constants differ"):
            ProtectedPublicSplitPolicy.from_dict(policy)

    def test_yt_unknown_floor_is_derived_from_zero_event_target(self) -> None:
        policy = ProtectedPublicSplitPolicy()
        self.assertEqual(policy.yt_test_unknown_minimum_identities, 299)
        self.assertEqual(
            policy.yt_test_unknown_minimum_identities,
            required_zero_event_trials(
                policy.yt_test_unknown_target_fpir,
                confidence_level=policy.yt_test_unknown_confidence_level,
            ),
        )

if __name__ == "__main__":
    unittest.main()

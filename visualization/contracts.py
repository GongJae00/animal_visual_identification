"""Strict normalized inputs consumed by every visualization recipe."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Self

from shared.foundation.provenance import canonical_json_bytes, content_sha256
from visualization.privacy import PublicationScope, validate_publishable_value

FIGURE_DATA_SCHEMA = "visualization.figure_data.v1"
FIGURE_DATA_BUNDLE_SCHEMA = "visualization.figure_data.bundle.v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")


class FigureContractError(ValueError):
    """Raised when normalized figure data violates its exact contract."""


@dataclass(frozen=True, slots=True)
class SourceBinding:
    """Content binding for one report, manifest, or declared source."""

    source_id: str
    schema_version: str
    content_sha256: str

    def __post_init__(self) -> None:
        _nonempty_text(self.source_id, "source_id")
        _nonempty_text(self.schema_version, "source schema_version")
        if not _SHA256.fullmatch(self.content_sha256):
            raise FigureContractError("source content_sha256 must be lowercase SHA-256")
        validate_publishable_value(
            {"source_id": self.source_id, "schema_version": self.schema_version},
            PublicationScope.PUBLIC,
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "source_id": self.source_id,
            "schema_version": self.schema_version,
            "content_sha256": self.content_sha256,
        }

    @classmethod
    def from_dict(cls, value: Any) -> Self:
        payload = _object(value, "source binding")
        _exact_keys(
            payload,
            {"source_id", "schema_version", "content_sha256"},
            "source binding",
        )
        return cls(
            source_id=payload["source_id"],
            schema_version=payload["schema_version"],
            content_sha256=payload["content_sha256"],
        )


@dataclass(frozen=True, slots=True)
class FigureData:
    """Immutable, content-normalized input for one registered figure."""

    figure_id: str
    kind: str
    scope: PublicationScope
    title: str
    caption: str
    limitations: tuple[str, ...]
    source_bindings: tuple[SourceBinding, ...]
    _payload_bytes: bytes

    @property
    def payload(self) -> dict[str, Any]:
        value = json.loads(self._payload_bytes)
        if not isinstance(value, dict):
            raise FigureContractError("figure payload must be an object")
        return value

    @classmethod
    def create(
        cls,
        *,
        figure_id: str,
        kind: str,
        scope: PublicationScope,
        title: str,
        caption: str,
        limitations: Sequence[str],
        source_bindings: Sequence[SourceBinding],
        payload: Mapping[str, Any],
    ) -> Self:
        _nonempty_text(figure_id, "figure_id")
        _nonempty_text(kind, "kind")
        if not isinstance(scope, PublicationScope):
            raise FigureContractError("scope must be a PublicationScope")
        _nonempty_text(title, "title")
        _nonempty_text(caption, "caption")
        parsed_limitations = _text_tuple(limitations, "limitations")
        if not parsed_limitations:
            raise FigureContractError("limitations must not be empty")
        parsed_bindings = tuple(source_bindings)
        if not parsed_bindings or not all(
            isinstance(binding, SourceBinding) for binding in parsed_bindings
        ):
            raise FigureContractError(
                "source_bindings must contain SourceBinding values"
            )
        source_ids = tuple(binding.source_id for binding in parsed_bindings)
        if len(source_ids) != len(set(source_ids)):
            raise FigureContractError("source binding IDs must be unique")
        normalized_payload = _object(payload, "payload")
        _validate_json_tree(normalized_payload)
        publishable = {
            "figure_id": figure_id,
            "kind": kind,
            "title": title,
            "caption": caption,
            "limitations": list(parsed_limitations),
            "source_bindings": [binding.to_dict() for binding in parsed_bindings],
            "payload": normalized_payload,
        }
        validate_publishable_value(publishable, scope)
        return cls(
            figure_id=figure_id,
            kind=kind,
            scope=scope,
            title=title,
            caption=caption,
            limitations=parsed_limitations,
            source_bindings=parsed_bindings,
            _payload_bytes=canonical_json_bytes(normalized_payload),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": FIGURE_DATA_SCHEMA,
            "figure_id": self.figure_id,
            "kind": self.kind,
            "scope": self.scope.value,
            "title": self.title,
            "caption": self.caption,
            "limitations": list(self.limitations),
            "source_bindings": [binding.to_dict() for binding in self.source_bindings],
            "payload": self.payload,
        }

    def to_bundle(self) -> dict[str, Any]:
        figure_data = self.to_dict()
        return {
            "schema_version": FIGURE_DATA_BUNDLE_SCHEMA,
            "figure_data": figure_data,
            "figure_data_sha256": content_sha256(figure_data),
        }

    @classmethod
    def from_dict(cls, value: Any) -> Self:
        payload = _object(value, "figure data")
        _exact_keys(
            payload,
            {
                "schema_version",
                "figure_id",
                "kind",
                "scope",
                "title",
                "caption",
                "limitations",
                "source_bindings",
                "payload",
            },
            "figure data",
        )
        if payload["schema_version"] != FIGURE_DATA_SCHEMA:
            raise FigureContractError("unsupported figure data schema")
        try:
            scope = PublicationScope(payload["scope"])
        except (TypeError, ValueError) as exc:
            raise FigureContractError("unsupported figure publication scope") from exc
        bindings = payload["source_bindings"]
        if not isinstance(bindings, list):
            raise FigureContractError("source_bindings must be an array")
        return cls.create(
            figure_id=payload["figure_id"],
            kind=payload["kind"],
            scope=scope,
            title=payload["title"],
            caption=payload["caption"],
            limitations=payload["limitations"],
            source_bindings=[SourceBinding.from_dict(item) for item in bindings],
            payload=payload["payload"],
        )

    @classmethod
    def from_bundle(cls, value: Any) -> Self:
        bundle = _object(value, "figure data bundle")
        _exact_keys(
            bundle,
            {"schema_version", "figure_data", "figure_data_sha256"},
            "figure data bundle",
        )
        if bundle["schema_version"] != FIGURE_DATA_BUNDLE_SCHEMA:
            raise FigureContractError("unsupported figure data bundle schema")
        digest = bundle["figure_data_sha256"]
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise FigureContractError("figure_data_sha256 must be lowercase SHA-256")
        if content_sha256(bundle["figure_data"]) != digest:
            raise FigureContractError(
                "figure data hash differs; input was tampered with"
            )
        return cls.from_dict(bundle["figure_data"])


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise FigureContractError(f"{name} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise FigureContractError(f"{name} keys must be strings")
    return dict(value)


def _exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise FigureContractError(f"{name} fields differ")


def _nonempty_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 512:
        raise FigureContractError(
            f"{name} must be non-empty text of at most 512 characters"
        )
    return value


def _text_tuple(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise FigureContractError(f"{name} must be an array")
    return tuple(_nonempty_text(item, name) for item in value)


def _validate_json_tree(root: Any) -> None:
    stack = [(root, 1)]
    nodes = 0
    while stack:
        value, depth = stack.pop()
        nodes += 1
        if nodes > 100_000 or depth > 24:
            raise FigureContractError("figure payload exceeds structural limits")
        if value is None or isinstance(value, (str, bool, int)):
            continue
        if isinstance(value, float):
            if not (-float("inf") < value < float("inf")):
                raise FigureContractError("figure payload numbers must be finite")
            continue
        if isinstance(value, Mapping):
            if not all(isinstance(key, str) for key in value):
                raise FigureContractError("figure payload keys must be strings")
            stack.extend((child, depth + 1) for child in value.values())
            continue
        if isinstance(value, (list, tuple)):
            stack.extend((child, depth + 1) for child in value)
            continue
        raise FigureContractError("figure payload must contain only JSON values")

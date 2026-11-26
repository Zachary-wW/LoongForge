"""Immutable, stdlib-only behavioral specifications shared by Intake and Anatomy."""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
from dataclasses import asdict, dataclass, fields


def _sha256(value: str) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None and value != "0" * 64


@dataclass(frozen=True, slots=True)
class ActionSpec:
    representation: str = "continuous"
    head: str = "flow_matching"
    horizon: int = 2
    dimension: int = 4
    units: str = "radians"
    coordinate_mode: str = "delta"
    normalization: str = "identity"
    vocabulary_size: int | None = None
    label_shift: int = 0
    tokenizer_id: str | None = None
    codebook_id: str | None = None
    codebook_sha256: str | None = None
    bos_token_id: int | None = None
    eos_token_id: int | None = None
    pad_token_id: int | None = None
    token_layout: tuple[str, ...] = ("batch", "horizon", "dimension")

    def __post_init__(self) -> None:
        if (self.representation, self.head) not in {
            ("continuous", "flow_matching"), ("discrete_tokens", "autoregressive"),
        }:
            raise ValueError("unsupported action representation/head")
        if any(type(x) is not int or x < 1 for x in (self.horizon, self.dimension)):
            raise ValueError("horizon and dimension must be positive integers")
        if self.coordinate_mode not in {"absolute", "delta"}:
            raise ValueError("coordinate_mode must be absolute or delta")
        if not all(isinstance(x, str) and x.strip() for x in (self.units, self.normalization)):
            raise ValueError("units and normalization must be explicit")
        if type(self.token_layout) is not tuple or any(not isinstance(x, str) for x in self.token_layout) or self.token_layout not in {
            ("batch", "horizon", "dimension"), ("batch", "dimension", "horizon"),
        }:
            raise ValueError("token_layout must declare batch-first BHD or BDH axes")
        if self.representation == "discrete_tokens":
            if type(self.vocabulary_size) is not int or self.vocabulary_size < 4:
                raise ValueError("discrete actions require vocabulary_size >= 4")
            if type(self.label_shift) is not int or not 1 <= self.label_shift < self.horizon * self.dimension:
                raise ValueError("discrete actions require an explicit positive label_shift")
            if self.horizon * self.dimension < 3:
                raise ValueError("token sequence requires BOS, payload and EOS positions")
            if not all(isinstance(x, str) and x.strip() for x in (self.tokenizer_id, self.codebook_id)):
                raise ValueError("discrete actions require tokenizer/codebook identifiers")
            if not _sha256(self.codebook_sha256):
                raise ValueError("discrete actions require a non-placeholder codebook SHA")
            special = (self.bos_token_id, self.eos_token_id, self.pad_token_id)
            if (any(type(x) is not int or not 0 <= x < self.vocabulary_size for x in special)
                    or len(set(special)) != 3):
                raise ValueError("BOS/EOS/PAD IDs must be distinct vocabulary entries")
        elif (self.vocabulary_size is not None or self.label_shift != 0
              or any(x is not None for x in (
                  self.tokenizer_id, self.codebook_id, self.codebook_sha256,
                  self.bos_token_id, self.eos_token_id, self.pad_token_id,
              ))):
            raise ValueError("continuous actions cannot declare discrete token semantics")

    def coordinates(self, batch_size: int) -> list[list[int]]:
        """Declared flattening order, each item expressed in canonical B/H/D coordinates."""
        sizes = {"batch": batch_size, "horizon": self.horizon, "dimension": self.dimension}
        return [
            [dict(zip(self.token_layout, values))[axis] for axis in ("batch", "horizon", "dimension")]
            for values in itertools.product(*(range(sizes[axis]) for axis in self.token_layout))
        ]

    def shifted_labels(self, tokens: list[int]) -> list[int]:
        """Shift independently within each batch sequence; never wrap into another sample."""
        length = self.horizon * self.dimension
        return [
            tokens[start + index + self.label_shift] if index + self.label_shift < length else self.pad_token_id
            for start in range(0, len(tokens), length) for index in range(length)
        ]


@dataclass(frozen=True, slots=True)
class ViewSpec:
    role: str
    required: bool = True
    enabled: bool = True
    shape: tuple[int, ...] = (3, 224, 224)  # excludes the case's batch axis
    dtype: str = "float32"
    mask: bool = True
    placeholder: str = "none"
    content_rule: str = "role_samples_sha256"

    def __post_init__(self) -> None:
        if not isinstance(self.role, str) or not self.role.strip():
            raise ValueError("view role must be non-empty")
        if any(type(x) is not bool for x in (self.required, self.enabled, self.mask)):
            raise ValueError("view required/enabled/mask flags must be booleans")
        if self.required and not self.enabled:
            raise ValueError("a required view cannot be disabled")
        if type(self.shape) is not tuple or len(self.shape) != 3 or any(type(x) is not int or x < 1 for x in self.shape):
            raise ValueError("view shape must be a positive immutable CHW tuple")
        if not isinstance(self.dtype, str) or self.dtype not in {"uint8", "float16", "bfloat16", "float32"}:
            raise ValueError("unsupported view dtype")
        if (not isinstance(self.placeholder, str) or self.placeholder not in {"none", "zeros"} or
                (self.mask and self.placeholder != "none") or
                (not self.mask and self.placeholder != "zeros")):
            raise ValueError("masked-out views require explicit zero placeholders")
        if not isinstance(self.content_rule, str) or self.content_rule not in {"sha256", "role_samples_sha256"}:
            raise ValueError("unknown view content digest rule")

    def content_digest(self, sample_ids: list[str]) -> str:
        if self.placeholder == "zeros":
            count = len(sample_ids) * math.prod(self.shape) * {"uint8": 1, "float16": 2, "bfloat16": 2, "float32": 4}[self.dtype]
            digest = hashlib.sha256()
            block = bytes(65536)
            for _ in range(count // len(block)):
                digest.update(block)
            digest.update(block[:count % len(block)])
            return digest.hexdigest()
        return hashlib.sha256(json.dumps({
            "role": self.role, "sample_ids": sample_ids, "shape": self.shape, "dtype": self.dtype,
        }, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class AnatomySpec:
    spec_id: str = "pi05"
    schema_version: int = 2
    required_camera_roles: tuple[str, ...] = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    optional_camera_roles: tuple[str, ...] = ()
    required_rng_consumers: tuple[str, ...] = ("flow_noise", "flow_time")
    action: ActionSpec = ActionSpec()
    views: tuple[ViewSpec, ...] = ()

    def __post_init__(self) -> None:
        if (not isinstance(self.spec_id, str) or not self.spec_id.strip()
                or type(self.schema_version) is not int or self.schema_version != 2):
            raise ValueError("spec_id and schema_version=2 are required")
        for items in (self.required_camera_roles, self.optional_camera_roles, self.required_rng_consumers):
            if (type(items) is not tuple or any(not isinstance(x, str) or not x.strip() for x in items)
                    or len(items) != len(set(items))):
                raise ValueError("spec roles/consumers must be unique immutable names")
        if set(self.required_camera_roles) & set(self.optional_camera_roles):
            raise ValueError("required and optional cameras must be disjoint")
        if not isinstance(self.action, ActionSpec):
            raise ValueError("action must be an ActionSpec")
        if type(self.views) is not tuple or any(not isinstance(x, ViewSpec) for x in self.views):
            raise ValueError("views must be immutable ViewSpec entries")
        if not self.views and (self.required_camera_roles or self.optional_camera_roles):
            # Compatibility convenience for constructors; all concrete defaults are serialized BEFORE capture.
            defaults = []
            for role in self.required_camera_roles + self.optional_camera_roles:
                required = role in self.required_camera_roles
                masked = self.spec_id == "pi05" and role == "right_wrist_0_rgb"
                defaults.append(ViewSpec(
                    role=role, required=required, enabled=required, mask=not masked,
                    placeholder="zeros" if masked else "none",
                ))
            object.__setattr__(self, "views", tuple(defaults))
        if (tuple(x.role for x in self.views if x.required) != self.required_camera_roles
                or tuple(x.role for x in self.views if not x.required) != self.optional_camera_roles):
            raise ValueError("view declarations and required/optional role lists disagree")

    def to_dict(self) -> dict:
        return json.loads(json.dumps(asdict(self)))

    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, doc: dict) -> AnatomySpec:
        if not isinstance(doc, dict) or set(doc) != {f.name for f in fields(cls)}:
            raise ValueError("AnatomySpec requires exactly its declared fields")
        values = dict(doc)
        action = values["action"]
        if not isinstance(action, dict) or set(action) != {f.name for f in fields(ActionSpec)}:
            raise ValueError("ActionSpec requires exactly its declared fields")
        action = dict(action)
        if not isinstance(action["token_layout"], list):
            raise ValueError("token_layout must be a list")
        action["token_layout"] = tuple(action["token_layout"])
        values["action"] = ActionSpec(**action)
        for key in ("required_camera_roles", "optional_camera_roles", "required_rng_consumers"):
            if not isinstance(values[key], list):
                raise ValueError(f"{key} must be a list")
            values[key] = tuple(values[key])
        if not isinstance(values["views"], list):
            raise ValueError("views must be a list")
        if not values["views"] and (values["required_camera_roles"] or values["optional_camera_roles"]):
            raise ValueError("serialized specs must declare concrete per-view policies")
        views = []
        for view in values["views"]:
            if not isinstance(view, dict) or set(view) != {f.name for f in fields(ViewSpec)}:
                raise ValueError("ViewSpec requires exactly its declared fields")
            view = dict(view)
            if not isinstance(view["shape"], list):
                raise ValueError("view shape must be a list")
            view["shape"] = tuple(view["shape"])
            views.append(ViewSpec(**view))
        values["views"] = tuple(views)
        return cls(**values)


PI05_SPEC = AnatomySpec()

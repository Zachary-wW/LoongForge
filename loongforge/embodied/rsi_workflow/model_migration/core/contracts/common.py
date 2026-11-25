from typing import Annotated, Any, Literal, TypeAlias

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
)

from .errors import ContractParseError


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        str_strip_whitespace=True,
    )


AssetKind: TypeAlias = Literal[
    "checkpoint",
    "dataset",
    "tokenizer",
    "statistics",
    "simulator",
]
AssetMode: TypeAlias = Literal["manifest_only", "owner_provided"]
ScopeKind: TypeAlias = Literal["f1_startup_probe", "f2_train_step"]


def _reject_zero_hash(value: str) -> str:
    if value == "0" * len(value):
        raise ValueError("all-zero hash is a placeholder")
    return value


Sha256: TypeAlias = Annotated[
    str,
    StringConstraints(pattern=r"^[0-9a-f]{64}$"),
    AfterValidator(_reject_zero_hash),
]
CommitHash: TypeAlias = Annotated[
    str,
    StringConstraints(pattern=r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$"),
    AfterValidator(_reject_zero_hash),
]


class AssetSpec(StrictModel):
    id: str = Field(min_length=1)
    kind: AssetKind
    required_for: list[ScopeKind] = Field(min_length=1)
    source: str | None = None
    revision: str | None = None
    expected_sha256: Sha256 | None = None
    access_policy: str = Field(min_length=1)
    local_path: str | None = None


def load_yaml_document(path: Any) -> dict[str, Any]:
    import yaml

    class UniqueKeyLoader(yaml.SafeLoader):
        pass

    def construct_mapping(loader: Any, node: Any, deep: bool = False) -> dict[Any, Any]:
        mapping: dict[Any, Any] = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in mapping:
                raise ContractParseError(
                    "DUPLICATE_YAML_KEY",
                    f"duplicate YAML key: {key}",
                    path=str(key),
                )
            mapping[key] = loader.construct_object(value_node, deep=deep)
        return mapping

    UniqueKeyLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping,
    )

    try:
        with open(path, encoding="utf-8") as stream:
            document = yaml.load(stream, Loader=UniqueKeyLoader)
    except OSError as error:
        raise ContractParseError(
            "CONTRACT_READ_FAILED",
            str(error),
            path=str(path),
        ) from error
    except yaml.YAMLError as error:
        raise ContractParseError(
            "YAML_PARSE_FAILED",
            str(error),
        ) from error
    if not isinstance(document, dict):
        raise ContractParseError(
            "CONTRACT_NOT_MAPPING",
            "contract document must be a mapping",
        )
    return document

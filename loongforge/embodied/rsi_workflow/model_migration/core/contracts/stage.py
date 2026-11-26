"""Stage specification contracts for the 6-stage pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, TypeAlias

from pydantic import Field, model_validator

from .common import CommitHash, StrictModel, load_yaml_document
from .errors import ContractValidationError

StageName: TypeAlias = Literal[
    "intake",
    "anatomy",
    "scaffold",
    "adapt",
    "verify",
    "deliver",
]


class StageSpec(StrictModel):
    """Declarative specification for one pipeline stage node."""

    schema_version: Literal[1] = 1
    id: str = Field(min_length=1)
    stage: StageName
    depends_on: list[str] = Field(default_factory=list)
    description: str = ""
    isolation: Literal["none", "read_only_reference", "git_worktree"] = "none"
    skills: list[str] = Field(default_factory=list)
    extension_points: list[str] = Field(default_factory=list)
    checks: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_unique_lists(self) -> StageSpec:
        if len(self.depends_on) != len(set(self.depends_on)):
            raise ValueError("stage depends_on entries must be unique")
        if self.id in self.depends_on:
            raise ValueError("stage cannot depend on itself")
        if len(self.skills) != len(set(self.skills)):
            raise ValueError("stage skills must be unique")
        if len(self.extension_points) != len(set(self.extension_points)):
            raise ValueError("stage extension_points must be unique")
        if len(self.checks) != len(set(self.checks)):
            raise ValueError("stage checks must be unique")
        if len(self.tools) != len(set(self.tools)):
            raise ValueError("stage tools must be unique")
        return self


PathBase: TypeAlias = Literal["workspace_root", "repo_root"]


class TargetFrameworkSpec(StrictModel):
    repo: str = Field(min_length=1)
    reference_commit: CommitHash
    commit_display: str | None = None
    ci_suite_command: str = Field(min_length=1)


class SkillsSourceSpec(StrictModel):
    repo: str = Field(min_length=1)
    commit: CommitHash
    path_base: PathBase
    root_relative_to_workspace: str = Field(min_length=1)


class ProfileSkillEntry(StrictModel):
    name: str = Field(min_length=1)
    path_base: PathBase
    path: str | None = None
    local_mirror_base: PathBase | None = None
    local_mirror: str | None = None
    optional: bool = False
    stages: list[StageName] = Field(min_length=1)
    role: str = Field(min_length=1)

    @model_validator(mode="after")
    def required_skill_must_declare_path(self) -> ProfileSkillEntry:
        if not self.optional and (self.path is None or not self.path):
            raise ValueError(
                f"non-optional skill {self.name!r} must declare a non-empty path"
            )
        return self


class ProfileStatelessToolEntry(StrictModel):
    id: str = Field(min_length=1)
    filename: str = Field(min_length=1)
    skill: str = Field(min_length=1)
    path_base: PathBase
    path: str = Field(min_length=1)
    stages: list[StageName] = Field(min_length=1)
    purpose: str = Field(min_length=1)


class PipelineProfile(StrictModel):
    """Validated top-level pipeline profile (e.g. `core/profiles/loongforge_embodied.yaml`)."""

    schema_version: Literal[1]
    profile_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    target_framework: TargetFrameworkSpec
    stages: list[StageSpec] = Field(min_length=1)
    extension_points: dict[str, dict[str, Any]] = Field(min_length=1)
    skills_source: SkillsSourceSpec
    skills: list[ProfileSkillEntry] = Field(min_length=1)
    optional_post_delivery_skills: list[str] = Field(default_factory=list)
    stateless_tools: list[ProfileStatelessToolEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_cross_references(self) -> PipelineProfile:
        stage_ids = [item.id for item in self.stages]
        if len(stage_ids) != len(set(stage_ids)):
            raise ValueError("profile stage ids must be unique")
        skill_names = [item.name for item in self.skills]
        if len(skill_names) != len(set(skill_names)):
            raise ValueError("profile skill names must be unique")
        known_skills = set(skill_names)
        known_ext = set(self.extension_points.keys())
        for stage in self.stages:
            unknown_skills = set(stage.skills) - known_skills
            if unknown_skills:
                raise ValueError(
                    f"stage {stage.id!r} references unknown skills: {sorted(unknown_skills)}"
                )
            unknown_ext = set(stage.extension_points) - known_ext
            if unknown_ext:
                raise ValueError(
                    f"stage {stage.id!r} references unknown extension_points: {sorted(unknown_ext)}"
                )
        for tool in self.stateless_tools:
            if tool.skill not in known_skills:
                raise ValueError(
                    f"stateless tool {tool.id!r} references unknown skill {tool.skill!r}"
                )
        return self

    def resolve_path(
        self,
        relative_path: str,
        *,
        path_base: PathBase,
        repo_root: Path,
        workspace_root: Path | None = None,
    ) -> Path:
        base = repo_root if path_base == "repo_root" else (workspace_root or repo_root.parent)
        return (base / relative_path).resolve()


def load_pipeline_profile(
    path: Any,
    *,
    verify_paths: bool = False,
    repo_root: Path | None = None,
    workspace_root: Path | None = None,
) -> PipelineProfile:
    """Load and strictly validate a pipeline profile YAML (rejecting duplicate keys)."""
    try:
        profile = PipelineProfile.model_validate(load_yaml_document(path))
    except ContractValidationError:
        raise
    except Exception as error:
        validation_error = getattr(error, "errors", None)
        if callable(validation_error):
            raise ContractValidationError(validation_error()) from error
        raise

    if verify_paths:
        resolved_repo = (
            repo_root.resolve()
            if repo_root is not None
            else Path(path).resolve().parents[2]
        )
        resolved_ws = (
            workspace_root.resolve()
            if workspace_root is not None
            else resolved_repo.parent
        )
        missing: list[dict[str, Any]] = []
        for skill in profile.skills:
            if skill.optional and skill.path is None:
                continue
            if skill.path is None:
                missing.append(
                    {
                        "loc": ("skills", skill.name, "path"),
                        "type": "missing_path",
                        "msg": f"required skill has no path: {skill.name}",
                    }
                )
                continue
            target = profile.resolve_path(
                skill.path,
                path_base=skill.path_base,
                repo_root=resolved_repo,
                workspace_root=resolved_ws,
            )
            if not target.exists():
                if skill.optional:
                    continue
                missing.append(
                    {
                        "loc": ("skills", skill.name, "path"),
                        "type": "missing_path",
                        "msg": f"skill path does not exist: {skill.path}",
                    }
                )
        for tool in profile.stateless_tools:
            target = profile.resolve_path(
                tool.path,
                path_base=tool.path_base,
                repo_root=resolved_repo,
                workspace_root=resolved_ws,
            )
            if not target.exists():
                missing.append(
                    {
                        "loc": ("stateless_tools", tool.id, "path"),
                        "type": "missing_path",
                        "msg": f"stateless tool path does not exist: {tool.path}",
                    }
                )
        if missing:
            raise ContractValidationError(missing)
    return profile


__all__ = [
    "PathBase",
    "PipelineProfile",
    "ProfileSkillEntry",
    "ProfileStatelessToolEntry",
    "SkillsSourceSpec",
    "StageName",
    "StageSpec",
    "TargetFrameworkSpec",
    "load_pipeline_profile",
]

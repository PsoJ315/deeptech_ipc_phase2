from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ProjectPaths:
    root: Path
    data_raw: Path
    data_interim: Path
    data_processed: Path
    artifacts: Path
    reports: Path


@dataclass
class PipelineConfig:
    project_name: str
    input_fields: list[str]
    support_fields: list[str]
    output_dir: str
    raw_file: str | None = None


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_paths() -> ProjectPaths:
    root = get_project_root()
    return ProjectPaths(
        root=root,
        data_raw=root / "data" / "raw",
        data_interim=root / "data" / "interim",
        data_processed=root / "data" / "processed",
        artifacts=root / "artifacts",
        reports=root / "reports",
    )


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def load_pipeline_config(config_path: Path | None = None) -> PipelineConfig:
    paths = get_paths()
    config_path = config_path or (paths.root / "configs" / "phase2.yaml")
    data = load_yaml(config_path)

    return PipelineConfig(
        project_name=data.get("project_name", "deeptech_ipc_phase2"),
        input_fields=list(data.get("input_fields", ["title", "abstract"])),
        support_fields=list(data.get("support_fields", ["claims"])),
        output_dir=data.get("output_dir", "artifacts"),
        raw_file=data.get("raw_file"),
    )
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from strux.settings import get_settings


def _config_path(filename: str) -> Path:
    return get_settings().project_root / "config" / filename


@lru_cache
def load_yaml_file(filename: str) -> dict[str, Any]:
    with _config_path(filename).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_skill_bridges() -> list[dict[str, Any]]:
    return load_yaml_file("skill_bridges.yaml").get("skills", [])


def load_major_profiles() -> list[dict[str, Any]]:
    return load_yaml_file("major_profiles.yaml").get("majors", [])


def load_activity_profiles() -> list[dict[str, Any]]:
    return load_yaml_file("activity_profiles.yaml").get("activities", [])


def load_korean_semantic_lexicon() -> list[dict[str, Any]]:
    return load_yaml_file("korean_semantic_lexicon.yaml").get("entries", [])


def load_market_signal_rules() -> list[dict[str, Any]]:
    return load_yaml_file("market_signal_rules.yaml").get("rules", [])

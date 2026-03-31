from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="STRUX_",
        extra="ignore",
        populate_by_name=True,
    )

    project_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[2])
    data_root: Path | None = None
    cluster_link_threshold: float = 0.12
    major_link_threshold: float = 0.18
    activity_link_threshold: float = 0.15
    top_skill_limit: int = 12
    openai_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("STRUX_OPENAI_API_KEY", "OPENAI_API_KEY"),
    )
    openai_model: str = "gpt-5-mini"
    openai_prompt_cache_key: str = "strux-profile-interpretation-v1"
    openai_max_output_tokens: int = 800
    openai_direct_match_threshold: float = 0.95
    openai_max_free_text_chars: int = 1_200
    openai_max_list_items_per_field: int = 12
    openai_max_item_chars: int = 120

    @property
    def resolved_data_root(self) -> Path:
        return self.data_root or self.project_root

    @property
    def openai_enabled(self) -> bool:
        return bool(self.openai_api_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()

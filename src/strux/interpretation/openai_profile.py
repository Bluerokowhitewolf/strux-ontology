from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from strux.interpretation.models import (
    InterpretationEvidence,
    InterpretationSourceField,
    ModelInterpretationResponse,
    ProfileInterpretationResult,
    ResolvedInterpretation,
)
from strux.interpretation.normalization import clamp, normalize_label, soft_match_score
from strux.ontology.graph import OntologyGraph
from strux.ontology.model import Entity, EntityType
from strux.settings import Settings, get_settings


FIELD_LABELS: dict[InterpretationSourceField, str] = {
    "free_text": "자유 서술 입력",
    "skills": "보유 역량",
    "activities": "활동 경험",
    "career_interests": "관심 진로",
    "major_interests": "관심 전공",
    "traits": "성향 단서",
}

FREE_TEXT_ENTITY_TYPES = (
    EntityType.SKILL,
    EntityType.ACTIVITY,
    EntityType.CAREER,
    EntityType.MAJOR,
    EntityType.TRAIT,
)


SYSTEM_PROMPT = """You are the interpretation layer for an ontology-driven student career decision support system.

Your job is to normalize multilingual student inputs into short English labels that can be resolved to ontology entities.

Rules:
- Do not recommend careers. Only normalize and extract ontology-ready labels.
- Output English labels only.
- Prefer concise canonical labels such as Computer Science, Programming, Critical Thinking, Research Project, Investigative.
- For explicit fields such as skills, activities, careers, majors, and traits, usually return one normalized label per raw input.
- For free_text, extract only the strongest ontology signals actually supported by the sentence.
- Do not invent unsupported entities.
- Keep reasoning short and concrete.
- Omit low-confidence guesses instead of forcing coverage.

Examples:
- "컴퓨터공학" in a major field -> "Computer Science"
- "데이터 분석 프로젝트" in an activity field -> "Data Analysis Project"
- "수학적 사고로 알고리즘 문제를 해결하는 것이 즐겁다" in free_text -> Mathematics, Programming, Critical Thinking
"""


@dataclass(frozen=True)
class _WorkItem:
    source_field: InterpretationSourceField
    raw_text: str
    allowed_entity_types: tuple[EntityType, ...]
    level_hint: float | None = None


class OpenAIProfileInterpreter:
    def __init__(
        self,
        graph: OntologyGraph,
        *,
        settings: Settings | None = None,
        client: Any | None = None,
    ) -> None:
        self.graph = graph
        self.settings = settings or get_settings()
        self._client = client

    def interpret_profile(self, profile: dict[str, Any]) -> ProfileInterpretationResult:
        work_items = self._collect_work_items(profile)
        local_resolutions: list[ResolvedInterpretation] = []
        pending_items: list[_WorkItem] = []

        for item in work_items:
            local_resolution = self._resolve_locally(item)
            if local_resolution is not None:
                local_resolutions.append(local_resolution)
                continue
            pending_items.append(item)

        warnings: list[str] = []
        openai_resolutions: list[ResolvedInterpretation] = []
        interpreter_mode = "local_exact_only"
        model_name: str | None = None

        if pending_items:
            if not self.settings.openai_enabled:
                warnings.append(
                    "OpenAI API 키가 없어 자유 서술과 비정형 입력은 해석하지 않았습니다. "
                    "영문 표준 엔티티와 직접 일치한 입력만 사용했습니다."
                )
            else:
                interpreter_mode = "openai_hybrid"
                model_name = self.settings.openai_model
                try:
                    model_response = self._run_model(profile, pending_items)
                except Exception as exc:
                    warnings.append(
                        "OpenAI 해석 호출에 실패해 직접 일치한 입력만 사용했습니다: "
                        f"{exc}"
                    )
                    interpreter_mode = "local_exact_only"
                else:
                    openai_resolutions = self._resolve_model_output(model_response)

        entities = self._dedupe_resolutions([*local_resolutions, *openai_resolutions])
        return ProfileInterpretationResult(
            strategy=(
                "Hybrid Mapping: local exact match + OpenAI semantic normalization + "
                "ontology entity resolution"
            ),
            interpreter_mode=interpreter_mode,
            model_name=model_name,
            warnings=warnings,
            entities=entities,
        )

    def _collect_work_items(self, profile: dict[str, Any]) -> list[_WorkItem]:
        items: list[_WorkItem] = []

        free_text = self._trim_text(profile.get("free_text"), self.settings.openai_max_free_text_chars)
        if free_text:
            items.append(
                _WorkItem(
                    source_field="free_text",
                    raw_text=free_text,
                    allowed_entity_types=FREE_TEXT_ENTITY_TYPES,
                )
            )

        for name, level in self._skill_entries(profile.get("skills", [])):
            if name:
                items.append(
                    _WorkItem(
                        source_field="skills",
                        raw_text=name,
                        allowed_entity_types=(EntityType.SKILL,),
                        level_hint=level,
                    )
                )

        for field_name, entity_type in (
            ("activities", EntityType.ACTIVITY),
            ("career_interests", EntityType.CAREER),
            ("major_interests", EntityType.MAJOR),
            ("traits", EntityType.TRAIT),
        ):
            for raw_text in self._unique_strings(profile.get(field_name, [])):
                items.append(
                    _WorkItem(
                        source_field=field_name,
                        raw_text=raw_text,
                        allowed_entity_types=(entity_type,),
                    )
                )

        return items

    def _skill_entries(self, values: Any) -> list[tuple[str, float]]:
        entries: list[tuple[str, float]] = []
        for raw in values[: self.settings.openai_max_list_items_per_field]:
            if isinstance(raw, dict):
                name = self._trim_text(raw.get("name"), self.settings.openai_max_item_chars)
                try:
                    level = clamp(float(raw.get("level", 0.7)))
                except (TypeError, ValueError):
                    level = 0.7
            else:
                name = self._trim_text(raw, self.settings.openai_max_item_chars)
                level = 0.7
            if name:
                entries.append((name, level))
        return entries

    def _unique_strings(self, values: Any) -> list[str]:
        cleaned: list[str] = []
        seen: set[str] = set()
        for raw in values[: self.settings.openai_max_list_items_per_field]:
            text = self._trim_text(raw, self.settings.openai_max_item_chars)
            normalized = normalize_label(text)
            if not text or normalized in seen:
                continue
            seen.add(normalized)
            cleaned.append(text)
        return cleaned

    def _trim_text(self, value: Any, max_chars: int) -> str:
        text = " ".join(str(value or "").split()).strip()
        if not text:
            return ""
        return text[:max_chars]

    def _resolve_locally(self, item: _WorkItem) -> ResolvedInterpretation | None:
        if item.source_field == "free_text":
            return None

        entity_type = item.allowed_entity_types[0]
        match = self.graph.search(entity_type, item.raw_text, limit=1)
        if not match:
            return None
        entity, match_score = match[0]
        if match_score < self.settings.openai_direct_match_threshold:
            return None

        return ResolvedInterpretation(
            entity_id=entity.id,
            entity_name=entity.name,
            entity_type=entity.type,
            score=round(match_score, 4),
            source_field=item.source_field,
            raw_text=item.raw_text,
            normalized_label=entity.name,
            direct=True,
            reasoning=(
                f"'{item.raw_text}' 입력이 온톨로지 표준명 '{entity.name}'와 직접 일치해 "
                "추가 모델 호출 없이 연결했습니다."
            ),
            activation_path=[
                f"{FIELD_LABELS[item.source_field]} -> {item.raw_text} -> 온톨로지 직접 일치 -> {entity.name}"
            ],
            evidence=[
                InterpretationEvidence(
                    cue=item.raw_text,
                    score=round(match_score, 4),
                    source="local-exact-match",
                    rationale="입력 표현이 온톨로지 표준명 또는 별칭과 직접 일치했습니다.",
                )
            ],
        )

    def _run_model(
        self,
        profile: dict[str, Any],
        pending_items: list[_WorkItem],
    ) -> ModelInterpretationResponse:
        client = self._get_client()
        payload = {
            "student_name": profile.get("name", "Anonymous Student"),
            "items": [
                {
                    "source_field": item.source_field,
                    "raw_text": item.raw_text,
                    "allowed_entity_types": [entity_type.value for entity_type in item.allowed_entity_types],
                }
                for item in pending_items
            ],
        }

        response = client.responses.parse(
            model=self.settings.openai_model,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            text_format=ModelInterpretationResponse,
            max_output_tokens=self.settings.openai_max_output_tokens,
            prompt_cache_key=self.settings.openai_prompt_cache_key,
        )
        return response.output_parsed or ModelInterpretationResponse()

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self.settings.openai_enabled:
            raise RuntimeError("OpenAI API key is not configured.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The 'openai' package is not installed. Install it with `pip install openai`."
            ) from exc
        self._client = OpenAI(api_key=self.settings.openai_api_key)
        return self._client

    def _resolve_model_output(
        self,
        model_response: ModelInterpretationResponse,
    ) -> list[ResolvedInterpretation]:
        resolutions: list[ResolvedInterpretation] = []
        for item in model_response.resolutions:
            entity = self._resolve_entity_label(item.entity_type, item.english_label)
            if entity is None:
                continue
            match_score = self._entity_label_score(entity, item.english_label)
            final_score = clamp((item.confidence * 0.75) + (match_score * 0.25))
            resolutions.append(
                ResolvedInterpretation(
                    entity_id=entity.id,
                    entity_name=entity.name,
                    entity_type=entity.type,
                    score=round(final_score, 4),
                    source_field=item.source_field,
                    raw_text=item.raw_text,
                    normalized_label=item.english_label,
                    direct=True,
                    reasoning=(
                        f"'{item.raw_text}'를 '{item.english_label}'로 정규화했고, "
                        f"온톨로지의 {entity.name} 엔티티로 연결했습니다."
                    ),
                    activation_path=[
                        (
                            f"{FIELD_LABELS[item.source_field]} -> {item.raw_text} -> GPT 정규화 -> "
                            f"{item.english_label} -> {entity.name}"
                        )
                    ],
                    evidence=[
                        InterpretationEvidence(
                            cue=item.evidence_span or item.raw_text,
                            score=round(item.confidence, 4),
                            source="openai-normalization",
                            rationale=item.reasoning,
                        ),
                        InterpretationEvidence(
                            cue=item.english_label,
                            score=round(match_score, 4),
                            source="ontology-resolution",
                            rationale="정규화된 영문 라벨을 온톨로지 표준 엔티티에 매핑했습니다.",
                        ),
                    ],
                )
            )
        return resolutions

    def _resolve_entity_label(self, entity_type: EntityType, label: str) -> Entity | None:
        matches = self.graph.search(entity_type, label, limit=1)
        if not matches:
            return None
        entity, score = matches[0]
        if score < 0.55:
            return None
        return entity

    def _entity_label_score(self, entity: Entity, label: str) -> float:
        candidates = [entity.name, *entity.aliases]
        return max((soft_match_score(candidate, label) for candidate in candidates), default=0.0)

    def _dedupe_resolutions(
        self,
        resolutions: list[ResolvedInterpretation],
    ) -> list[ResolvedInterpretation]:
        merged: dict[tuple[str, str, str], ResolvedInterpretation] = {}
        for item in resolutions:
            key = (item.entity_id, item.source_field, normalize_label(item.raw_text))
            existing = merged.get(key)
            if existing is None:
                merged[key] = item
                continue
            existing.score = max(existing.score, item.score)
            if item.normalized_label and item.normalized_label != existing.normalized_label:
                existing.normalized_label = item.normalized_label
            existing.evidence.extend(item.evidence)
            for path in item.activation_path:
                if path not in existing.activation_path:
                    existing.activation_path.append(path)

        ordered = sorted(
            merged.values(),
            key=lambda item: (item.score, item.entity_type.value, item.entity_name),
            reverse=True,
        )
        return ordered[:24]


def interpret_profile_input(
    profile: dict[str, Any],
    graph: OntologyGraph | None = None,
    *,
    settings: Settings | None = None,
) -> ProfileInterpretationResult:
    if graph is None:
        from strux.runtime import get_runtime

        graph = get_runtime().graph
    interpreter = OpenAIProfileInterpreter(graph, settings=settings)
    return interpreter.interpret_profile(profile)

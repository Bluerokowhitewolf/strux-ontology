from types import SimpleNamespace

from strux.interpretation.models import (
    ModelInterpretationResolution,
    ModelInterpretationResponse,
)
from strux.interpretation.openai_profile import OpenAIProfileInterpreter
from strux.ontology.model import EntityType
from strux.runtime import get_runtime
from strux.settings import Settings


class _FakeResponses:
    def parse(self, **_: object) -> SimpleNamespace:
        return SimpleNamespace(
            output_parsed=ModelInterpretationResponse(
                resolutions=[
                    ModelInterpretationResolution(
                        source_field="free_text",
                        raw_text="수학적 사고로 알고리즘 문제를 해결하는 것이 즐겁다",
                        entity_type=EntityType.SKILL,
                        english_label="Mathematics",
                        confidence=0.93,
                        reasoning="The sentence explicitly signals mathematical thinking.",
                        evidence_span="수학적 사고",
                    ),
                    ModelInterpretationResolution(
                        source_field="free_text",
                        raw_text="수학적 사고로 알고리즘 문제를 해결하는 것이 즐겁다",
                        entity_type=EntityType.SKILL,
                        english_label="Programming",
                        confidence=0.9,
                        reasoning="Algorithm problem solving aligns with programming skill.",
                        evidence_span="알고리즘 문제",
                    ),
                    ModelInterpretationResolution(
                        source_field="free_text",
                        raw_text="수학적 사고로 알고리즘 문제를 해결하는 것이 즐겁다",
                        entity_type=EntityType.SKILL,
                        english_label="Critical Thinking",
                        confidence=0.88,
                        reasoning="Problem solving suggests critical thinking.",
                        evidence_span="문제를 해결",
                    ),
                ]
            )
        )


class _FakeClient:
    def __init__(self) -> None:
        self.responses = _FakeResponses()


def test_openai_profile_interpreter_resolves_foundational_skills() -> None:
    interpreter = OpenAIProfileInterpreter(
        get_runtime().graph,
        settings=Settings(openai_api_key="test-key"),
        client=_FakeClient(),
    )

    result = interpreter.interpret_profile(
        {"free_text": "수학적 사고로 알고리즘 문제를 해결하는 것이 즐겁다"}
    )

    skills = {
        item.entity_name: item
        for item in result.entities
        if item.entity_type == EntityType.SKILL
    }

    assert result.interpreter_mode == "openai_hybrid"
    assert "Mathematics" in skills
    assert "Programming" in skills
    assert "Critical Thinking" in skills
    assert skills["Mathematics"].normalized_label == "Mathematics"
    assert any(evidence.source == "openai-normalization" for evidence in skills["Programming"].evidence)

from typing import Literal

from pydantic import BaseModel, Field

from strux.ontology.model import EntityType


InterpretationSourceField = Literal[
    "free_text",
    "skills",
    "activities",
    "career_interests",
    "major_interests",
    "traits",
]


class InterpretationEvidence(BaseModel):
    cue: str
    score: float
    source: str
    rationale: str


class ResolvedInterpretation(BaseModel):
    entity_id: str
    entity_name: str
    entity_type: EntityType
    score: float
    source_field: InterpretationSourceField
    raw_text: str
    normalized_label: str | None = None
    direct: bool = True
    reasoning: str
    activation_path: list[str] = Field(default_factory=list)
    evidence: list[InterpretationEvidence] = Field(default_factory=list)


class ModelInterpretationResolution(BaseModel):
    source_field: InterpretationSourceField
    raw_text: str
    entity_type: EntityType
    english_label: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    evidence_span: str


class ModelInterpretationResponse(BaseModel):
    resolutions: list[ModelInterpretationResolution] = Field(default_factory=list)


class ProfileInterpretationResult(BaseModel):
    strategy: str
    interpreter_mode: str
    model_name: str | None = None
    warnings: list[str] = Field(default_factory=list)
    entities: list[ResolvedInterpretation] = Field(default_factory=list)

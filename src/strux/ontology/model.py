from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class EvidenceLayer(StrEnum):
    RAW_DATA = "raw-data"
    INTERPRETATION = "interpretation"
    ONTOLOGY = "ontology"
    DECISION_SUPPORT = "decision-support"


class EntityType(StrEnum):
    STUDENT = "student"
    CAREER = "career"
    SKILL = "skill"
    MAJOR = "major"
    ACTIVITY = "activity"
    TRAIT = "trait"
    KNOWLEDGE = "knowledge"


class RelationType(StrEnum):
    HAS_SKILL = "hasSkill"
    HAS_INTEREST_IN = "hasInterestIn"
    HAS_TRAIT = "hasTrait"
    REQUIRES_SKILL = "requiresSkill"
    REQUIRES_KNOWLEDGE = "requiresKnowledge"
    ALIGNED_WITH_TRAIT = "alignedWithTrait"
    INVOLVES_ACTIVITY = "involvesActivity"
    RELATED_TO_CAREER = "relatedToCareer"
    SPECIALIZES_TO = "specializesTo"
    DEVELOPS_SKILL = "developsSkill"
    LEADS_TO = "leadsTo"
    SUPPORTS_MAJOR = "supportsMajor"
    SUPPORTS_CAREER = "supportsCareer"
    HAS_PREREQUISITE = "hasPrerequisite"


class Evidence(BaseModel):
    source: str
    layer: EvidenceLayer
    note: str | None = None
    locator: str | None = None
    weight: float | None = None


class Entity(BaseModel):
    id: str
    type: EntityType
    name: str
    description: str | None = None
    aliases: list[str] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)
    provenance: list[Evidence] = Field(default_factory=list)


class Relation(BaseModel):
    subject_id: str
    predicate: RelationType
    object_id: str
    weight: float = 1.0
    attributes: dict[str, Any] = Field(default_factory=dict)
    evidence: list[Evidence] = Field(default_factory=list)

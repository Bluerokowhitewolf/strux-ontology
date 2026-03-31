from pydantic import BaseModel, Field, model_validator

from strux.interpretation.models import ResolvedInterpretation


class StudentSkillInput(BaseModel):
    name: str
    level: float = Field(default=0.7, ge=0.0, le=1.0)

    @model_validator(mode="before")
    @classmethod
    def coerce_from_string(cls, value: object) -> object:
        if isinstance(value, str):
            return {"name": value, "level": 0.7}
        return value


class StudentProfile(BaseModel):
    name: str = "Anonymous Student"
    free_text: str = ""
    skills: list[StudentSkillInput] = Field(default_factory=list)
    activities: list[str] = Field(default_factory=list)
    career_interests: list[str] = Field(default_factory=list)
    major_interests: list[str] = Field(default_factory=list)
    traits: list[str] = Field(default_factory=list)
    preferred_job_zones: list[int] = Field(default_factory=list)
    top_k: int = Field(default=5, ge=1, le=10)


class SkillSignal(BaseModel):
    skill_name: str
    score: float
    source: str


class InterpretedStudentState(BaseModel):
    normalized_skills: list[SkillSignal]
    inferred_skills: list[SkillSignal]
    profile_interpretations: list[ResolvedInterpretation] = Field(default_factory=list)
    free_text_interpretations: list[ResolvedInterpretation] = Field(default_factory=list)
    matched_activities: list[str]
    matched_career_interests: list[str]
    matched_major_interests: list[str]
    matched_traits: list[str]
    interpretation_mode: str
    interpretation_warnings: list[str] = Field(default_factory=list)


class ScoreComponent(BaseModel):
    label: str
    score: float


class ReadinessZoneSignal(BaseModel):
    job_zone: int
    zone_label: str
    score: float
    best_matching_career: str | None = None
    evidence_skills: list[str] = Field(default_factory=list)
    bridge_skills: list[str] = Field(default_factory=list)
    explanation: str


class StudentReadinessSummary(BaseModel):
    current_job_zone: int
    current_zone_label: str
    next_job_zone: int | None = None
    next_zone_label: str | None = None
    zone_scores: list[ReadinessZoneSignal] = Field(default_factory=list)
    priority_bridge_skills: list[str] = Field(default_factory=list)
    explanation: str


class ReadinessPlan(BaseModel):
    current_job_zone: int
    current_zone_label: str
    target_job_zone: int | None = None
    target_zone_label: str | None = None
    readiness_gap: int | None = None
    status: str
    target_reference: list[str] = Field(default_factory=list)
    bridge_skills: list[str] = Field(default_factory=list)
    recommended_majors: list[str] = Field(default_factory=list)
    suggested_activities: list[str] = Field(default_factory=list)
    milestone_path: list[str] = Field(default_factory=list)
    explanation: str


class MarketSignalSummary(BaseModel):
    score: float
    market_contexts: list[str] = Field(default_factory=list)
    sample_titles: list[str] = Field(default_factory=list)
    technology_highlights: list[str] = Field(default_factory=list)
    matched_signal_skills: list[str] = Field(default_factory=list)
    bridge_signal_skills: list[str] = Field(default_factory=list)
    explanation: str


class CareerRecommendation(BaseModel):
    career_id: str
    career_name: str
    score: float
    job_zone: int | None = None
    fit_label: str
    matched_skills: list[str]
    missing_skills: list[str]
    matched_traits: list[str]
    recommended_knowledge_areas: list[str]
    career_contexts: list[str]
    supporting_majors: list[str]
    suggested_activities: list[str]
    score_breakdown: list[ScoreComponent]
    readiness_plan: ReadinessPlan
    market_signal: MarketSignalSummary
    next_actions: list[str]
    ontology_paths: list[str]
    explanation: str


class MajorRecommendation(BaseModel):
    major_id: str
    major_name: str
    score: float
    key_skills: list[str]
    connected_careers: list[str]
    explanation: str


class ActivityRecommendation(BaseModel):
    activity_id: str
    activity_name: str
    score: float
    focus_skills: list[str]
    supports_targets: list[str]
    explanation: str


class RecommendationReport(BaseModel):
    student_name: str
    student_summary: str
    decision_principle: str
    interpreted_state: InterpretedStudentState
    student_readiness: StudentReadinessSummary
    top_careers: list[CareerRecommendation]
    top_majors: list[MajorRecommendation]
    top_activities: list[ActivityRecommendation]
    epistemic_note: str


class WorkspaceNode(BaseModel):
    id: str
    label: str
    entity_type: str
    column: int
    score: float | None = None
    status: str = "default"
    meta: dict[str, object] = Field(default_factory=dict)


class WorkspaceEdge(BaseModel):
    source: str
    target: str
    relation: str
    weight: float = 1.0
    status: str = "default"


class OntologyWorkspace(BaseModel):
    nodes: list[WorkspaceNode] = Field(default_factory=list)
    edges: list[WorkspaceEdge] = Field(default_factory=list)
    default_focus_node_id: str | None = None


class WorkspaceAnalysis(BaseModel):
    report: RecommendationReport
    workspace: OntologyWorkspace


class ActionSimulationRequest(BaseModel):
    profile: StudentProfile
    activity_name: str


class CareerImpact(BaseModel):
    career_id: str
    career_name: str
    baseline_score: float
    simulated_score: float
    score_delta: float
    score_delta_percent: float
    baseline_rank: int | None = None
    simulated_rank: int | None = None
    rank_change: int | None = None
    explanation: str


class SkillImpact(BaseModel):
    skill_name: str
    baseline_score: float
    simulated_score: float
    score_delta: float
    explanation: str


class ActionSimulationReport(BaseModel):
    student_name: str
    injected_activity_id: str
    injected_activity_name: str
    activity_resolution_reasoning: str
    already_present: bool = False
    action_summary: str
    future_feedback: list[str] = Field(default_factory=list)
    career_impacts: list[CareerImpact] = Field(default_factory=list)
    skill_impacts: list[SkillImpact] = Field(default_factory=list)
    baseline_report: RecommendationReport
    simulated_report: RecommendationReport

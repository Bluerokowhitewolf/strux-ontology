from collections import defaultdict
from dataclasses import dataclass, field

from strux.config.profiles import load_market_signal_rules
from strux.decision.models import (
    ActivityRecommendation,
    ActionSimulationReport,
    CareerImpact,
    CareerRecommendation,
    InterpretedStudentState,
    MajorRecommendation,
    MarketSignalSummary,
    OntologyWorkspace,
    ReadinessPlan,
    ReadinessZoneSignal,
    RecommendationReport,
    SkillSignal,
    SkillImpact,
    StudentProfile,
    StudentReadinessSummary,
    WorkspaceAnalysis,
    WorkspaceEdge,
    WorkspaceNode,
)
from strux.decision.pathing import STUDENT_SOURCE_ID, BridgeAnalysis, BridgePathPlanner
from strux.interpretation.models import ResolvedInterpretation
from strux.interpretation.openai_profile import OpenAIProfileInterpreter
from strux.interpretation.normalization import (
    clamp,
    normalize_label,
    normalize_semantic_text,
    soft_match_score,
    weighted_overlap,
)
from strux.ontology.graph import OntologyGraph
from strux.ontology.model import Entity, EntityType, RelationType
from strux.ontology.semantics import entity_skill_vector
from strux.presentation.korean import component_breakdown, fit_label, job_zone_label


@dataclass
class StudentState:
    explicit_skills: dict[str, float] = field(default_factory=dict)
    inferred_skills: dict[str, float] = field(default_factory=dict)
    combined_skills: dict[str, float] = field(default_factory=dict)
    profile_resolutions: list[ResolvedInterpretation] = field(default_factory=list)
    free_text_resolutions: list[ResolvedInterpretation] = field(default_factory=list)
    matched_activities: list[Entity] = field(default_factory=list)
    matched_career_interests: list[Entity] = field(default_factory=list)
    matched_major_interests: list[Entity] = field(default_factory=list)
    matched_traits: list[Entity] = field(default_factory=list)
    skill_sources: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    interpretation_mode: str = "local_exact_only"
    interpretation_warnings: list[str] = field(default_factory=list)


@dataclass
class ZoneReadinessState:
    job_zone: int
    score: float
    direct_alignment: float
    bridge_analysis: BridgeAnalysis
    best_matching_career: Entity | None = None
    evidence_skills: list[str] = field(default_factory=list)
    bridge_skills: list[str] = field(default_factory=list)


@dataclass
class StudentReadinessState:
    current_job_zone: int
    zone_states: list[ZoneReadinessState] = field(default_factory=list)
    next_job_zone: int | None = None
    priority_bridge_skills: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class MarketRule:
    name: str
    patterns: tuple[str, ...]
    skill_weights: dict[str, float]


class RecommendationEngine:
    def __init__(
        self,
        graph: OntologyGraph,
        *,
        profile_interpreter: OpenAIProfileInterpreter | None = None,
    ) -> None:
        self.graph = graph
        self.profile_interpreter = profile_interpreter or OpenAIProfileInterpreter(graph)
        self.occupations = [
            entity
            for entity in self.graph.entities_by_type(EntityType.CAREER)
            if entity.attributes.get("profile_kind") == "occupation"
        ]
        self.majors = self.graph.entities_by_type(EntityType.MAJOR)
        self.student_activities = [
            entity
            for entity in self.graph.entities_by_type(EntityType.ACTIVITY)
            if entity.attributes.get("activity_kind") == "student"
        ]
        self.bridge_planner = BridgePathPlanner(self.graph)
        self.occupation_skill_vectors = {
            career.id: entity_skill_vector(
                self.graph,
                career.id,
                (RelationType.REQUIRES_SKILL,),
                include_prerequisites=False,
            )
            for career in self.occupations
        }
        self.zone_profiles = self._build_zone_profiles()
        self.zone_careers = self._group_careers_by_zone()
        self.market_rules = self._load_market_signal_rules()

    def recommend(self, profile: StudentProfile) -> RecommendationReport:
        state = self._interpret_student(profile)
        return self._build_report(profile, state)

    def analyze_workspace(self, profile: StudentProfile) -> WorkspaceAnalysis:
        state = self._interpret_student(profile)
        report = self._build_report(profile, state)
        workspace = self._build_workspace(profile, state, report)
        return WorkspaceAnalysis(report=report, workspace=workspace)

    def simulate_activity_injection(
        self,
        profile: StudentProfile,
        activity_name: str,
    ) -> ActionSimulationReport:
        baseline_state = self._interpret_student(profile)
        baseline_report = self._build_report(profile, baseline_state)

        activity, resolution_reasoning = self._resolve_simulated_activity(activity_name)
        already_present = any(
            normalize_label(existing) == normalize_label(activity.name)
            for existing in profile.activities
        )

        simulated_profile = profile.model_copy(deep=True)
        if not already_present:
            simulated_profile.activities.append(activity.name)

        simulated_state = self._interpret_student(simulated_profile)
        simulated_report = self._build_report(simulated_profile, simulated_state)

        career_impacts = self._career_impacts(
            activity=activity,
            baseline_report=baseline_report,
            simulated_report=simulated_report,
        )
        skill_impacts = self._skill_impacts(
            activity=activity,
            baseline_state=baseline_state,
            simulated_state=simulated_state,
        )

        return ActionSimulationReport(
            student_name=profile.name,
            injected_activity_id=activity.id,
            injected_activity_name=activity.name,
            activity_resolution_reasoning=resolution_reasoning,
            already_present=already_present,
            action_summary=self._action_summary(activity.name, already_present, career_impacts, skill_impacts),
            future_feedback=self._future_feedback(activity.name, already_present, career_impacts, skill_impacts),
            career_impacts=career_impacts[:5],
            skill_impacts=skill_impacts[:6],
            baseline_report=baseline_report,
            simulated_report=simulated_report,
        )

    def _build_report(self, profile: StudentProfile, state: StudentState) -> RecommendationReport:
        readiness = self._infer_student_readiness(state)
        top_careers = self._recommend_careers(profile, state, readiness)
        top_majors = self._recommend_majors(profile, state, top_careers)
        top_activities = self._recommend_activities(profile, state, top_careers, top_majors)
        return RecommendationReport(
            student_name=profile.name,
            student_summary=self._student_summary(profile, state, top_careers, readiness),
            decision_principle=(
                "이 시스템은 원자료를 그대로 추천하지 않고, 학생·진로·역량·전공·활동 간 관계를 "
                "해석한 뒤 근거가 보이는 형태로 의사결정을 지원합니다."
            ),
            interpreted_state=self._build_interpreted_state(state),
            student_readiness=self._build_readiness_summary(readiness),
            top_careers=top_careers,
            top_majors=top_majors,
            top_activities=top_activities,
            epistemic_note=(
                "이 결과는 절대적 판정이 아니라, 현재 입력과 데이터 구조를 바탕으로 구성한 "
                "운영상의 의미 해석 결과입니다."
            ),
        )

    def _interpret_student(self, profile: StudentProfile) -> StudentState:
        state = StudentState()
        interpretation = self.profile_interpreter.interpret_profile(profile.model_dump(mode="python"))
        state.profile_resolutions = interpretation.entities
        state.free_text_resolutions = [
            item for item in interpretation.entities if item.source_field == "free_text"
        ]
        state.interpretation_mode = interpretation.interpreter_mode
        state.interpretation_warnings = interpretation.warnings

        skill_levels = self._skill_level_map(profile)
        for resolution in interpretation.entities:
            entity = self.graph.entity(resolution.entity_id)
            if resolution.source_field == "skills" and entity.type == EntityType.SKILL:
                declared_level = skill_levels.get(
                    self._skill_level_key(resolution.raw_text),
                    0.7,
                )
                state.explicit_skills[entity.id] = max(
                    state.explicit_skills.get(entity.id, 0.0),
                    declared_level,
                )
                state.skill_sources[entity.id].append(
                    f"보유역량:{resolution.normalized_label or entity.name}"
                )
                continue

            if resolution.source_field == "activities" and entity.type == EntityType.ACTIVITY:
                self._register_activity(
                    state,
                    entity,
                    strength=max(0.85, resolution.score),
                    source_label=f"활동경험:{resolution.normalized_label or entity.name}",
                )
                continue

            if resolution.source_field == "career_interests" and entity.type == EntityType.CAREER:
                self._append_unique_entity(state.matched_career_interests, entity)
                continue

            if resolution.source_field == "major_interests" and entity.type == EntityType.MAJOR:
                self._append_unique_entity(state.matched_major_interests, entity)
                continue

            if resolution.source_field == "traits" and entity.type == EntityType.TRAIT:
                self._append_unique_entity(state.matched_traits, entity)
                continue

            if resolution.source_field != "free_text":
                continue

            if entity.type == EntityType.SKILL:
                state.inferred_skills[entity.id] = max(
                    state.inferred_skills.get(entity.id, 0.0),
                    resolution.score,
                )
                state.skill_sources[entity.id].append(f"자유서술:{entity.name}")
            elif entity.type == EntityType.ACTIVITY:
                self._register_activity(
                    state,
                    entity,
                    strength=resolution.score,
                    source_label=f"자유서술 활동:{entity.name}",
                )
            elif entity.type == EntityType.CAREER:
                self._append_unique_entity(state.matched_career_interests, entity)
            elif entity.type == EntityType.MAJOR:
                self._append_unique_entity(state.matched_major_interests, entity)
            elif entity.type == EntityType.TRAIT:
                self._append_unique_entity(state.matched_traits, entity)

        merged = dict(state.explicit_skills)
        for skill_id, score in state.inferred_skills.items():
            merged[skill_id] = max(merged.get(skill_id, 0.0), score)
        state.combined_skills = self._expand_skill_map(merged, state)
        return state

    def _recommend_careers(
        self,
        profile: StudentProfile,
        state: StudentState,
        readiness: StudentReadinessState,
    ) -> list[CareerRecommendation]:
        scored: list[tuple[Entity, float, dict[str, float], BridgeAnalysis, MarketSignalSummary]] = []
        student_skills = state.combined_skills

        for career in self.occupations:
            career_vector = self.occupation_skill_vectors[career.id]
            if not career_vector:
                continue
            skill_alignment = weighted_overlap(student_skills, career_vector)
            missing_vector = self._missing_skill_vector(student_skills, career_vector)
            bridge_analysis = self.bridge_planner.analyze(student_skills, missing_vector)
            if max(skill_alignment, bridge_analysis.bridge_reachability) < 0.05:
                continue

            market_signal = self._market_signal_summary(state, career)
            components = {
                "skill_alignment": skill_alignment,
                "bridge_reachability": bridge_analysis.bridge_reachability,
                "path_efficiency": bridge_analysis.path_efficiency,
                "frontier_signal": bridge_analysis.frontier_signal,
                "interest_alignment": self._career_interest_alignment(state, career),
                "activity_alignment": self._activity_alignment(state, career),
                "major_alignment": self._major_alignment(state, career),
                "trait_alignment": self._trait_alignment(state, career),
                "market_alignment": market_signal.score,
                "zone_fit": self._zone_fit(profile, career),
                "readiness_path": self._readiness_path_alignment(readiness, career, bridge_analysis),
            }
            total = clamp(
                (components["skill_alignment"] * 0.42)
                + (components["bridge_reachability"] * 0.10)
                + (components["path_efficiency"] * 0.06)
                + (components["frontier_signal"] * 0.03)
                + (components["interest_alignment"] * 0.10)
                + (components["activity_alignment"] * 0.10)
                + (components["major_alignment"] * 0.07)
                + (components["trait_alignment"] * 0.03)
                + (components["market_alignment"] * 0.02)
                + (components["zone_fit"] * 0.01)
                + (components["readiness_path"] * 0.06)
            )
            scored.append((career, total, components, bridge_analysis, market_signal))

        scored.sort(key=lambda item: item[1], reverse=True)

        recommendations: list[CareerRecommendation] = []
        for career, score, components, bridge_analysis, market_signal in scored[: profile.top_k]:
            career_vector = self.occupation_skill_vectors[career.id]
            matched_skills = self._top_matched_skills(state.combined_skills, career_vector)
            missing_skills = self._top_missing_skills(state.combined_skills, career_vector)
            matched_traits = self._matched_trait_names(state, career)
            knowledge_areas = self._top_knowledge_areas(career)
            contexts = self._career_contexts(career)
            supporting_majors = self._supporting_major_names(state, career, limit=3)
            suggested_activities = self._supporting_activity_names(state, career, limit=3)
            bridge_skills = self._bridge_skill_names(bridge_analysis, limit=3)
            readiness_plan = self._build_readiness_plan(
                state=state,
                readiness=readiness,
                career=career,
                bridge_analysis=bridge_analysis,
                supporting_majors=supporting_majors,
                suggested_activities=suggested_activities,
                missing_skills=missing_skills,
            )

            recommendations.append(
                CareerRecommendation(
                    career_id=career.id,
                    career_name=career.name,
                    score=round(score, 4),
                    job_zone=career.attributes.get("job_zone"),
                    fit_label=fit_label(score),
                    matched_skills=matched_skills,
                    missing_skills=missing_skills,
                    matched_traits=matched_traits,
                    recommended_knowledge_areas=knowledge_areas,
                    career_contexts=contexts,
                    supporting_majors=supporting_majors,
                    suggested_activities=suggested_activities,
                    score_breakdown=component_breakdown(components),
                    readiness_plan=readiness_plan,
                    market_signal=market_signal,
                    next_actions=self._build_next_actions(
                        market_signal=market_signal,
                        readiness_plan=readiness_plan,
                        bridge_skills=bridge_skills,
                        missing_skills=missing_skills,
                        suggested_activities=suggested_activities,
                        recommended_knowledge=knowledge_areas,
                        supporting_majors=supporting_majors,
                    ),
                    ontology_paths=self._build_paths(
                        career=career,
                        matched_skills=matched_skills,
                        majors=supporting_majors,
                        activities=suggested_activities,
                        bridge_analysis=bridge_analysis,
                    ),
                    explanation=self._career_explanation(
                        market_signal=market_signal,
                        readiness_plan=readiness_plan,
                        career=career,
                        matched_skills=matched_skills,
                        bridge_skills=bridge_skills,
                        matched_traits=matched_traits,
                        contexts=contexts,
                    ),
                )
            )
        return recommendations

    def _recommend_majors(
        self,
        profile: StudentProfile,
        state: StudentState,
        top_careers: list[CareerRecommendation],
    ) -> list[MajorRecommendation]:
        if not top_careers:
            return []

        target_vector = self._aggregate_target_vector([career.career_id for career in top_careers])
        student_vector = state.combined_skills
        target_career_ids = {career.career_id for career in top_careers}
        scored: list[tuple[Entity, float]] = []

        for major in self.majors:
            major_vector = entity_skill_vector(self.graph, major.id, (RelationType.DEVELOPS_SKILL,))
            support = weighted_overlap(major_vector, target_vector)
            gap_closure = weighted_overlap(major_vector, self._missing_skill_vector(student_vector, target_vector))
            career_bonus = max(
                (
                    relation.weight
                    for relation in self.graph.relations_from(major.id, RelationType.LEADS_TO)
                    if relation.object_id in target_career_ids
                ),
                default=0.0,
            )
            interest_bonus = 1.0 if any(match.id == major.id for match in state.matched_major_interests) else 0.0
            score = clamp((support * 0.45) + (gap_closure * 0.25) + (career_bonus * 0.20) + (interest_bonus * 0.10))
            if score >= 0.1:
                scored.append((major, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        return [
            MajorRecommendation(
                major_id=major.id,
                major_name=major.name,
                score=round(score, 4),
                key_skills=self._major_key_skills(major),
                connected_careers=self._major_connected_careers(major, top_careers),
                explanation="이 전공은 상위 진로와 직접 연결되며, 현재 부족한 역량을 메우는 데 유리합니다.",
            )
            for major, score in scored[:3]
        ]

    def _recommend_activities(
        self,
        profile: StudentProfile,
        state: StudentState,
        top_careers: list[CareerRecommendation],
        top_majors: list[MajorRecommendation],
    ) -> list[ActivityRecommendation]:
        if not top_careers:
            return []

        target_vector = self._aggregate_target_vector([career.career_id for career in top_careers])
        missing_vector = self._missing_skill_vector(state.combined_skills, target_vector)
        major_ids = {major.major_id for major in top_majors}
        career_ids = {career.career_id for career in top_careers}
        scored: list[tuple[Entity, float]] = []

        for activity in self.student_activities:
            activity_vector = entity_skill_vector(self.graph, activity.id, (RelationType.DEVELOPS_SKILL,))
            gap_support = weighted_overlap(activity_vector, missing_vector)
            major_bonus = max(
                (
                    relation.weight
                    for relation in self.graph.relations_from(activity.id, RelationType.SUPPORTS_MAJOR)
                    if relation.object_id in major_ids
                ),
                default=0.0,
            )
            career_bonus = max(
                (
                    relation.weight
                    for relation in self.graph.relations_from(activity.id, RelationType.SUPPORTS_CAREER)
                    if relation.object_id in career_ids
                ),
                default=0.0,
            )
            score = clamp((gap_support * 0.60) + (major_bonus * 0.15) + (career_bonus * 0.25))
            if score >= 0.1:
                scored.append((activity, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        return [
            ActivityRecommendation(
                activity_id=activity.id,
                activity_name=activity.name,
                score=round(score, 4),
                focus_skills=self._activity_focus_skills(activity, missing_vector),
                supports_targets=self._activity_support_targets(activity, top_careers, top_majors),
                explanation="이 활동은 부족한 역량을 채우면서 추천 전공·진로와 구조적으로 연결됩니다.",
            )
            for activity, score in scored[:3]
        ]

    def _build_interpreted_state(self, state: StudentState) -> InterpretedStudentState:
        return InterpretedStudentState(
            normalized_skills=self._skill_signals(state.explicit_skills, state, declared=True),
            inferred_skills=self._skill_signals(state.inferred_skills, state, declared=False),
            profile_interpretations=state.profile_resolutions[:16],
            free_text_interpretations=state.free_text_resolutions[:10],
            matched_activities=[entity.name for entity in state.matched_activities],
            matched_career_interests=[entity.name for entity in state.matched_career_interests],
            matched_major_interests=[entity.name for entity in state.matched_major_interests],
            matched_traits=[entity.name for entity in state.matched_traits],
            interpretation_mode=state.interpretation_mode,
            interpretation_warnings=state.interpretation_warnings,
        )

    def _build_workspace(
        self,
        profile: StudentProfile,
        state: StudentState,
        report: RecommendationReport,
    ) -> OntologyWorkspace:
        student_node_id = "workspace:student"
        nodes: dict[str, WorkspaceNode] = {}
        edges: dict[tuple[str, str, str], WorkspaceEdge] = {}

        def add_node(
            *,
            node_id: str,
            label: str,
            entity_type: str,
            column: int,
            score: float | None = None,
            status: str = "default",
            meta: dict[str, object] | None = None,
        ) -> None:
            current = nodes.get(node_id)
            payload = WorkspaceNode(
                id=node_id,
                label=label,
                entity_type=entity_type,
                column=column,
                score=round(score, 4) if isinstance(score, float) else score,
                status=status,
                meta=meta or {},
            )
            if current is None:
                nodes[node_id] = payload
                return
            merged_meta = {**current.meta, **payload.meta}
            nodes[node_id] = current.model_copy(
                update={
                    "column": min(current.column, payload.column),
                    "score": max(current.score or 0.0, payload.score or 0.0) if payload.score is not None else current.score,
                    "status": current.status if current.status.startswith("career") else payload.status,
                    "meta": merged_meta,
                }
            )

        def add_edge(
            *,
            source: str,
            target: str,
            relation: str,
            weight: float = 1.0,
            status: str = "default",
        ) -> None:
            key = (source, target, relation)
            current = edges.get(key)
            payload = WorkspaceEdge(
                source=source,
                target=target,
                relation=relation,
                weight=round(weight, 4),
                status=status,
            )
            if current is None:
                edges[key] = payload
                return
            edges[key] = current.model_copy(
                update={
                    "weight": max(current.weight, payload.weight),
                    "status": current.status if current.status != "default" else payload.status,
                }
            )

        add_node(
            node_id=student_node_id,
            label=profile.name or "학생",
            entity_type=EntityType.STUDENT.value,
            column=0,
            status="student",
            meta={
                "summary": report.student_summary,
                "readiness_zone": report.student_readiness.current_job_zone,
                "readiness_label": report.student_readiness.current_zone_label,
            },
        )

        for skill_id, score in sorted(state.explicit_skills.items(), key=lambda item: item[1], reverse=True)[:6]:
            skill = self.graph.entity(skill_id)
            add_node(
                node_id=skill.id,
                label=skill.name,
                entity_type=skill.type.value,
                column=1,
                score=score,
                status="declared_skill",
                meta={"source": "직접 입력"},
            )
            add_edge(
                source=student_node_id,
                target=skill.id,
                relation=RelationType.HAS_SKILL.value,
                weight=score,
                status="direct",
            )

        for skill_id, score in sorted(state.inferred_skills.items(), key=lambda item: item[1], reverse=True)[:4]:
            skill = self.graph.entity(skill_id)
            add_node(
                node_id=skill.id,
                label=skill.name,
                entity_type=skill.type.value,
                column=1,
                score=score,
                status="inferred_skill",
                meta={"source": "해석/활동 추론"},
            )
            add_edge(
                source=student_node_id,
                target=skill.id,
                relation="inferredSkill",
                weight=score,
                status="inferred",
            )

        for activity in state.matched_activities[:4]:
            add_node(
                node_id=activity.id,
                label=activity.name,
                entity_type=activity.type.value,
                column=1,
                status="experienced_activity",
                meta={"description": activity.description or ""},
            )
            add_edge(
                source=student_node_id,
                target=activity.id,
                relation="experiencedActivity",
                weight=1.0,
                status="experience",
            )
            for relation in self.graph.relations_from(activity.id, RelationType.DEVELOPS_SKILL)[:4]:
                if relation.object_id not in nodes:
                    continue
                add_edge(
                    source=activity.id,
                    target=relation.object_id,
                    relation=RelationType.DEVELOPS_SKILL.value,
                    weight=relation.weight,
                    status="support",
                )

        for trait in state.matched_traits[:3]:
            add_node(
                node_id=trait.id,
                label=trait.name,
                entity_type=trait.type.value,
                column=1,
                status="trait",
            )
            add_edge(
                source=student_node_id,
                target=trait.id,
                relation=RelationType.HAS_TRAIT.value,
                weight=1.0,
                status="trait",
            )

        visible_skill_ids = {node_id for node_id, node in nodes.items() if node.entity_type == EntityType.SKILL.value}
        visible_career_ids: set[str] = set()
        visible_major_ids: set[str] = set()
        visible_activity_ids: set[str] = set()

        for index, career in enumerate(report.top_careers[:3]):
            visible_career_ids.add(career.career_id)
            add_node(
                node_id=career.career_id,
                label=career.career_name,
                entity_type=EntityType.CAREER.value,
                column=4,
                score=career.score,
                status="career_primary" if index == 0 else "career_secondary",
                meta={
                    "job_zone": career.job_zone,
                    "fit_label": career.fit_label,
                    "readiness_gap": career.readiness_plan.readiness_gap,
                },
            )

            for skill_name in career.matched_skills[:3]:
                skill = self.graph.find_by_name(EntityType.SKILL, skill_name)
                if skill is None:
                    continue
                add_node(
                    node_id=skill.id,
                    label=skill.name,
                    entity_type=skill.type.value,
                    column=1,
                    status="matched_skill",
                )
                visible_skill_ids.add(skill.id)
                add_edge(
                    source=skill.id,
                    target=career.career_id,
                    relation=RelationType.REQUIRES_SKILL.value,
                    weight=0.95,
                    status="matched",
                )

            for skill_name in [*career.readiness_plan.bridge_skills[:3], *career.market_signal.bridge_signal_skills[:2]]:
                skill = self.graph.find_by_name(EntityType.SKILL, skill_name)
                if skill is None:
                    continue
                add_node(
                    node_id=skill.id,
                    label=skill.name,
                    entity_type=skill.type.value,
                    column=2,
                    status="bridge_skill",
                )
                visible_skill_ids.add(skill.id)
                add_edge(
                    source=skill.id,
                    target=career.career_id,
                    relation=RelationType.REQUIRES_SKILL.value,
                    weight=0.85,
                    status="bridge",
                )
                for relation in self.graph.relations_from(skill.id, RelationType.HAS_PREREQUISITE)[:2]:
                    prerequisite = self.graph.entity(relation.object_id)
                    add_node(
                        node_id=prerequisite.id,
                        label=prerequisite.name,
                        entity_type=prerequisite.type.value,
                        column=1,
                        status="prerequisite_skill",
                    )
                    visible_skill_ids.add(prerequisite.id)
                    add_edge(
                        source=prerequisite.id,
                        target=skill.id,
                        relation=RelationType.HAS_PREREQUISITE.value,
                        weight=relation.weight,
                        status="bridge",
                    )

            for skill_name in career.missing_skills[:2]:
                skill = self.graph.find_by_name(EntityType.SKILL, skill_name)
                if skill is None:
                    continue
                add_node(
                    node_id=skill.id,
                    label=skill.name,
                    entity_type=skill.type.value,
                    column=2,
                    status="gap_skill",
                )
                visible_skill_ids.add(skill.id)
                add_edge(
                    source=skill.id,
                    target=career.career_id,
                    relation=RelationType.REQUIRES_SKILL.value,
                    weight=0.7,
                    status="gap",
                )

        for major in report.top_majors[:3]:
            visible_major_ids.add(major.major_id)
            add_node(
                node_id=major.major_id,
                label=major.major_name,
                entity_type=EntityType.MAJOR.value,
                column=3,
                score=major.score,
                status="recommended_major",
            )
            for career_id in visible_career_ids:
                relations = self.graph.relations_from(major.major_id, RelationType.LEADS_TO)
                for relation in relations:
                    if relation.object_id != career_id:
                        continue
                    add_edge(
                        source=major.major_id,
                        target=career_id,
                        relation=RelationType.LEADS_TO.value,
                        weight=relation.weight,
                        status="support",
                    )
            for relation in self.graph.relations_from(major.major_id, RelationType.DEVELOPS_SKILL)[:5]:
                if relation.object_id not in visible_skill_ids:
                    continue
                add_edge(
                    source=major.major_id,
                    target=relation.object_id,
                    relation=RelationType.DEVELOPS_SKILL.value,
                    weight=relation.weight,
                    status="support",
                )

        for activity in report.top_activities[:3]:
            visible_activity_ids.add(activity.activity_id)
            add_node(
                node_id=activity.activity_id,
                label=activity.activity_name,
                entity_type=EntityType.ACTIVITY.value,
                column=3,
                score=activity.score,
                status="recommended_activity",
            )
            for relation in self.graph.relations_from(activity.activity_id, RelationType.SUPPORTS_CAREER):
                if relation.object_id not in visible_career_ids:
                    continue
                add_edge(
                    source=activity.activity_id,
                    target=relation.object_id,
                    relation=RelationType.SUPPORTS_CAREER.value,
                    weight=relation.weight,
                    status="support",
                )
            for relation in self.graph.relations_from(activity.activity_id, RelationType.SUPPORTS_MAJOR):
                if relation.object_id not in visible_major_ids:
                    continue
                add_edge(
                    source=activity.activity_id,
                    target=relation.object_id,
                    relation=RelationType.SUPPORTS_MAJOR.value,
                    weight=relation.weight,
                    status="support",
                )
            for relation in self.graph.relations_from(activity.activity_id, RelationType.DEVELOPS_SKILL)[:5]:
                if relation.object_id not in visible_skill_ids:
                    continue
                add_edge(
                    source=activity.activity_id,
                    target=relation.object_id,
                    relation=RelationType.DEVELOPS_SKILL.value,
                    weight=relation.weight,
                    status="support",
                )

        for interest in state.matched_career_interests[:2]:
            if interest.id not in visible_career_ids:
                continue
            add_edge(
                source=student_node_id,
                target=interest.id,
                relation=RelationType.HAS_INTEREST_IN.value,
                weight=1.0,
                status="interest",
            )

        for interest in state.matched_major_interests[:2]:
            if interest.id not in visible_major_ids:
                continue
            add_edge(
                source=student_node_id,
                target=interest.id,
                relation=RelationType.HAS_INTEREST_IN.value,
                weight=1.0,
                status="interest",
            )

        return OntologyWorkspace(
            nodes=sorted(nodes.values(), key=lambda item: (item.column, item.entity_type, -(item.score or 0.0), item.label)),
            edges=sorted(edges.values(), key=lambda item: (item.source, item.target, item.relation)),
            default_focus_node_id=student_node_id,
        )

    def _load_market_signal_rules(self) -> list[MarketRule]:
        rules: list[MarketRule] = []
        for spec in load_market_signal_rules():
            skill_weights: dict[str, float] = {}
            for skill_name, weight in spec.get("skills", {}).items():
                skill = self.graph.find_by_name(EntityType.SKILL, skill_name)
                if skill is None:
                    continue
                skill_weights[skill.id] = clamp(float(weight))
            patterns = tuple(
                pattern
                for raw_pattern in spec.get("patterns", [])
                if (pattern := normalize_semantic_text(raw_pattern))
            )
            if not patterns or not skill_weights:
                continue
            rules.append(
                MarketRule(
                    name=str(spec.get("name", "market-rule")),
                    patterns=patterns,
                    skill_weights=skill_weights,
                )
            )
        return rules

    def _group_careers_by_zone(self) -> dict[int, list[Entity]]:
        grouped: dict[int, list[Entity]] = defaultdict(list)
        for career in self.occupations:
            job_zone = career.attributes.get("job_zone")
            if isinstance(job_zone, int):
                grouped[job_zone].append(career)
        return dict(sorted(grouped.items()))

    def _market_signal_summary(
        self,
        state: StudentState,
        career: Entity,
    ) -> MarketSignalSummary:
        student_skills = state.combined_skills
        market_contexts, sample_titles, context_vector = self._career_market_context(career)
        technology_vector, technology_highlights = self._career_technology_signal_vector(career)

        market_target_vector = dict(context_vector)
        for skill_id, weight in technology_vector.items():
            market_target_vector[skill_id] = max(market_target_vector.get(skill_id, 0.0), weight)

        missing_market_vector = self._missing_skill_vector(student_skills, market_target_vector)
        bridge_analysis = self.bridge_planner.analyze(student_skills, missing_market_vector)
        context_alignment = weighted_overlap(student_skills, context_vector)
        technology_alignment = weighted_overlap(student_skills, technology_vector)
        demand_density = self._market_demand_density(career)
        score = clamp(
            (context_alignment * 0.35)
            + (technology_alignment * 0.22)
            + (bridge_analysis.bridge_reachability * 0.20)
            + (bridge_analysis.path_efficiency * 0.13)
            + (demand_density * 0.10)
        )

        matched_signal_skills = self._top_matched_skills(student_skills, market_target_vector, limit=4)
        bridge_signal_skills = self._readiness_bridge_skills(bridge_analysis, limit=4)
        if not bridge_signal_skills:
            bridge_signal_skills = self._top_missing_skills(student_skills, market_target_vector, limit=4)

        parts: list[str] = []
        if market_contexts:
            parts.append(f"시장 맥락은 {', '.join(market_contexts[:2])} 축으로 수렴합니다")
        if matched_signal_skills:
            parts.append(f"{', '.join(matched_signal_skills[:2])} 역량이 시장 신호와 이미 맞닿아 있습니다")
        if bridge_signal_skills:
            parts.append(f"{', '.join(bridge_signal_skills[:2])} 역량을 보완하면 시장 기술 요구에 더 가까워집니다")
        if technology_highlights:
            parts.append(f"대표 기술 사례는 {technology_highlights[0]} 입니다")

        return MarketSignalSummary(
            score=round(score, 4),
            market_contexts=market_contexts[:3],
            sample_titles=sample_titles[:4],
            technology_highlights=technology_highlights[:4],
            matched_signal_skills=matched_signal_skills,
            bridge_signal_skills=bridge_signal_skills,
            explanation=". ".join(parts) + "." if parts else "시장 신호 근거가 제한적이라 구조적 정합도를 우선 해석합니다.",
        )

    def _career_market_context(
        self,
        career: Entity,
    ) -> tuple[list[str], list[str], dict[str, float]]:
        contexts: list[tuple[str, float]] = []
        sample_titles: list[str] = []
        context_vector: dict[str, float] = {}
        for relation in sorted(
            self.graph.relations_to(career.id, RelationType.SPECIALIZES_TO),
            key=lambda item: item.weight,
            reverse=True,
        ):
            source = self.graph.entity(relation.subject_id)
            profile_kind = source.attributes.get("profile_kind")
            if profile_kind not in {"market_sector", "career_cluster"}:
                continue
            contexts.append((source.name, relation.weight))
            for title in source.attributes.get("sample_titles", []):
                if title not in sample_titles:
                    sample_titles.append(title)
            source_vector = entity_skill_vector(self.graph, source.id, (RelationType.REQUIRES_SKILL,))
            for skill_id, weight in source_vector.items():
                propagated = clamp(weight * relation.weight)
                context_vector[skill_id] = max(context_vector.get(skill_id, 0.0), propagated)

        ordered_contexts: list[str] = []
        seen_contexts: set[str] = set()
        for context_name, _ in contexts:
            if context_name in seen_contexts:
                continue
            seen_contexts.add(context_name)
            ordered_contexts.append(context_name)
        return ordered_contexts, sample_titles, context_vector

    def _career_technology_signal_vector(
        self,
        career: Entity,
    ) -> tuple[dict[str, float], list[str]]:
        vector: dict[str, float] = {}
        highlights: list[str] = []
        technology_examples = career.attributes.get("technology_examples") or []
        if not isinstance(technology_examples, list):
            return vector, highlights

        ranked_examples = sorted(
            technology_examples,
            key=lambda item: (bool(item.get("in_demand")), bool(item.get("hot_technology"))),
            reverse=True,
        )

        for item in ranked_examples[:8]:
            signal_text = normalize_semantic_text(
                f"{item.get('example', '')} {item.get('commodity_title', '')}"
            )
            if not signal_text:
                continue

            per_item_skills: dict[str, float] = {}
            base_weight = 0.30
            if item.get("hot_technology"):
                base_weight += 0.20
            if item.get("in_demand"):
                base_weight += 0.30

            for rule in self.market_rules:
                if not any(pattern in signal_text for pattern in rule.patterns):
                    continue
                for skill_id, rule_weight in rule.skill_weights.items():
                    resolved_weight = clamp(base_weight * rule_weight)
                    per_item_skills[skill_id] = max(per_item_skills.get(skill_id, 0.0), resolved_weight)
                    vector[skill_id] = max(vector.get(skill_id, 0.0), resolved_weight)

            if not per_item_skills:
                continue

            item_name = str(item.get("example") or item.get("commodity_title") or "").strip()
            if not item_name:
                continue
            tags: list[str] = []
            if item.get("in_demand"):
                tags.append("수요 높음")
            if item.get("hot_technology"):
                tags.append("핫 기술")
            linked_skill_names = [
                self.graph.entity(skill_id).name
                for skill_id, _ in sorted(per_item_skills.items(), key=lambda row: row[1], reverse=True)[:2]
            ]
            label = item_name
            if tags:
                label += f" ({', '.join(tags)})"
            if linked_skill_names:
                label += f" -> {', '.join(linked_skill_names)}"
            if label not in highlights:
                highlights.append(label)

        return vector, highlights

    def _market_demand_density(self, career: Entity) -> float:
        technology_examples = career.attributes.get("technology_examples") or []
        if not isinstance(technology_examples, list) or not technology_examples:
            return 0.0
        in_demand = sum(1 for item in technology_examples if item.get("in_demand"))
        hot = sum(1 for item in technology_examples if item.get("hot_technology"))
        return clamp(((in_demand / len(technology_examples)) * 0.7) + ((hot / len(technology_examples)) * 0.3))

    def _build_zone_profiles(self) -> dict[int, dict[str, float]]:
        weight_totals: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        frequency_totals: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        zone_sizes: dict[int, int] = defaultdict(int)

        for career in self.occupations:
            job_zone = career.attributes.get("job_zone")
            if not isinstance(job_zone, int):
                continue
            vector = self.occupation_skill_vectors.get(career.id, {})
            if not vector:
                continue
            zone_sizes[job_zone] += 1
            for skill_id, weight in vector.items():
                weight_totals[job_zone][skill_id] += weight
                frequency_totals[job_zone][skill_id] += 1

        profiles: dict[int, dict[str, float]] = {}
        for job_zone, totals in weight_totals.items():
            zone_size = max(zone_sizes[job_zone], 1)
            ranked: list[tuple[str, float]] = []
            for skill_id, total_weight in totals.items():
                frequency = frequency_totals[job_zone][skill_id] / zone_size
                mean_weight = total_weight / max(frequency_totals[job_zone][skill_id], 1)
                score = clamp((mean_weight * 0.70) + (frequency * 0.30))
                if score >= 0.18 or frequency >= 0.35:
                    ranked.append((skill_id, round(score, 4)))
            ranked.sort(key=lambda item: item[1], reverse=True)
            profiles[job_zone] = dict(ranked[:18])
        return dict(sorted(profiles.items()))

    def _infer_student_readiness(self, state: StudentState) -> StudentReadinessState:
        if not self.zone_careers:
            return StudentReadinessState(current_job_zone=1)

        zone_states: list[ZoneReadinessState] = []
        for job_zone, careers in self.zone_careers.items():
            ranked_zone_careers: list[tuple[float, float, BridgeAnalysis, Entity, dict[str, float]]] = []
            for career in careers:
                career_vector = self.occupation_skill_vectors.get(career.id, {})
                if not career_vector:
                    continue
                missing_vector = self._missing_skill_vector(state.combined_skills, career_vector)
                bridge_analysis = self.bridge_planner.analyze(state.combined_skills, missing_vector)
                direct_alignment = weighted_overlap(state.combined_skills, career_vector)
                structural_score = clamp(
                    (direct_alignment * 0.68)
                    + (bridge_analysis.bridge_reachability * 0.16)
                    + (bridge_analysis.path_efficiency * 0.10)
                    + (bridge_analysis.frontier_signal * 0.06)
                )
                ranked_zone_careers.append(
                    (structural_score, direct_alignment, bridge_analysis, career, career_vector)
                )

            if not ranked_zone_careers:
                continue

            ranked_zone_careers.sort(key=lambda item: item[0], reverse=True)
            top_scores = [item[0] for item in ranked_zone_careers[:3]]
            zone_score = clamp(
                (ranked_zone_careers[0][0] * 0.65)
                + ((sum(top_scores) / len(top_scores)) * 0.35)
            )
            _, direct_alignment, bridge_analysis, best_career, best_vector = ranked_zone_careers[0]
            zone_states.append(
                ZoneReadinessState(
                    job_zone=job_zone,
                    score=round(zone_score, 4),
                    direct_alignment=round(direct_alignment, 4),
                    bridge_analysis=bridge_analysis,
                    best_matching_career=best_career,
                    evidence_skills=self._top_matched_skills(state.combined_skills, best_vector, limit=3),
                    bridge_skills=self._readiness_bridge_skills(bridge_analysis, limit=3),
                )
            )

        current_job_zone = 1
        for zone_state in zone_states:
            threshold = self._readiness_threshold(zone_state.job_zone)
            if zone_state.score >= threshold:
                current_job_zone = zone_state.job_zone

        max_zone = max(self.zone_careers)
        next_job_zone = current_job_zone + 1 if current_job_zone < max_zone else None
        next_zone_state = next(
            (item for item in zone_states if item.job_zone == next_job_zone),
            None,
        )
        priority_bridge_skills = (
            next_zone_state.bridge_skills
            if next_zone_state is not None and next_zone_state.bridge_skills
            else next_zone_state.evidence_skills
            if next_zone_state is not None
            else []
        )
        return StudentReadinessState(
            current_job_zone=current_job_zone,
            zone_states=zone_states,
            next_job_zone=next_job_zone,
            priority_bridge_skills=priority_bridge_skills[:3],
        )

    def _build_readiness_summary(self, readiness: StudentReadinessState) -> StudentReadinessSummary:
        zone_scores: list[ReadinessZoneSignal] = []
        for zone_state in readiness.zone_states:
            best_matching_career = (
                zone_state.best_matching_career.name
                if zone_state.best_matching_career is not None
                else None
            )
            explanation_parts: list[str] = []
            if zone_state.evidence_skills:
                explanation_parts.append(
                    f"{', '.join(zone_state.evidence_skills[:2])} 역량이 이 준비도 층과 직접 겹칩니다"
                )
            if zone_state.bridge_skills:
                explanation_parts.append(
                    f"{', '.join(zone_state.bridge_skills[:2])} 역량이 다음 단계로 가는 브리지입니다"
                )
            if best_matching_career:
                explanation_parts.append(f"이 층에서 가장 가까운 직업 예시는 {best_matching_career} 입니다")
            zone_scores.append(
                ReadinessZoneSignal(
                    job_zone=zone_state.job_zone,
                    zone_label=job_zone_label(zone_state.job_zone),
                    score=zone_state.score,
                    best_matching_career=best_matching_career,
                    evidence_skills=zone_state.evidence_skills,
                    bridge_skills=zone_state.bridge_skills,
                    explanation=". ".join(explanation_parts) + "." if explanation_parts else "추가 구조 해석이 필요합니다.",
                )
            )

        current_zone_label = job_zone_label(readiness.current_job_zone)
        next_zone_label = (
            job_zone_label(readiness.next_job_zone) if readiness.next_job_zone is not None else None
        )
        if readiness.next_job_zone is not None and readiness.priority_bridge_skills:
            explanation = (
                f"현재 학생 상태는 Job Zone {readiness.current_job_zone}({current_zone_label})에 가장 가깝습니다. "
                f"다음 단계인 Job Zone {readiness.next_job_zone}({next_zone_label})로 가려면 "
                f"{', '.join(readiness.priority_bridge_skills[:2])} 브리지 역량을 우선 강화하는 경로가 유리합니다."
            )
        elif readiness.next_job_zone is not None:
            explanation = (
                f"현재 학생 상태는 Job Zone {readiness.current_job_zone}({current_zone_label})에 가장 가깝습니다. "
                f"다음 단계는 Job Zone {readiness.next_job_zone}({next_zone_label})입니다."
            )
        else:
            explanation = (
                f"현재 학생 상태는 Job Zone {readiness.current_job_zone}({current_zone_label})에 가깝고, "
                "상위 준비도는 개별 진로 경로를 기준으로 세밀하게 판단하는 편이 적절합니다."
            )

        return StudentReadinessSummary(
            current_job_zone=readiness.current_job_zone,
            current_zone_label=current_zone_label,
            next_job_zone=readiness.next_job_zone,
            next_zone_label=next_zone_label,
            zone_scores=zone_scores,
            priority_bridge_skills=readiness.priority_bridge_skills,
            explanation=explanation,
        )

    def _readiness_threshold(self, job_zone: int) -> float:
        return clamp(0.11 + ((job_zone - 1) * 0.045), lower=0.11, upper=0.32)

    def _readiness_bridge_skills(
        self,
        bridge_analysis: BridgeAnalysis,
        *,
        limit: int = 4,
    ) -> list[str]:
        names: list[str] = []
        for evidence in bridge_analysis.ranked_paths(limit=limit * 3):
            if evidence.reachability < 0.1:
                continue
            ordered_skill_ids = [*evidence.bridge_skill_ids, evidence.target_skill_id]
            for skill_id in ordered_skill_ids:
                skill_name = self.graph.entity(skill_id).name
                if skill_name in names:
                    continue
                names.append(skill_name)
                if len(names) >= limit:
                    return names
        return names

    def _readiness_path_alignment(
        self,
        readiness: StudentReadinessState,
        career: Entity,
        bridge_analysis: BridgeAnalysis,
    ) -> float:
        target_job_zone = career.attributes.get("job_zone")
        if not isinstance(target_job_zone, int):
            return clamp(bridge_analysis.path_efficiency * 0.25)

        target_zone_state = next(
            (item for item in readiness.zone_states if item.job_zone == target_job_zone),
            None,
        )
        target_zone_score = target_zone_state.score if target_zone_state is not None else 0.0

        if target_job_zone <= readiness.current_job_zone:
            return clamp((target_zone_score * 0.45) + (bridge_analysis.path_efficiency * 0.25) + 0.30)

        gap = target_job_zone - readiness.current_job_zone
        return clamp(
            (target_zone_score * 0.35)
            + (bridge_analysis.bridge_reachability * 0.30)
            + (bridge_analysis.path_efficiency * 0.20)
            + (bridge_analysis.frontier_signal * 0.10)
            + (max(0.0, 1.0 - (gap * 0.22)) * 0.05)
            - (max(gap - 1, 0) * 0.08)
        )

    def _build_readiness_plan(
        self,
        *,
        state: StudentState,
        readiness: StudentReadinessState,
        career: Entity,
        bridge_analysis: BridgeAnalysis,
        supporting_majors: list[str],
        suggested_activities: list[str],
        missing_skills: list[str],
    ) -> ReadinessPlan:
        current_job_zone = readiness.current_job_zone
        current_zone_label = job_zone_label(current_job_zone)
        target_job_zone = career.attributes.get("job_zone")
        target_zone_label = (
            job_zone_label(target_job_zone) if isinstance(target_job_zone, int) else None
        )
        readiness_gap = (
            max(target_job_zone - current_job_zone, 0)
            if isinstance(target_job_zone, int)
            else None
        )
        bridge_skills = self._readiness_bridge_skills(bridge_analysis, limit=4) or missing_skills[:4]
        focused_majors = self._readiness_major_names(state, career, bridge_analysis, limit=3)
        if not focused_majors:
            focused_majors = supporting_majors[:3]
        target_reference = self._job_zone_reference_lines(career)
        status = self._readiness_status(current_job_zone, target_job_zone, focused_majors)
        milestone_path = self._readiness_milestones(
            career=career,
            current_job_zone=current_job_zone,
            target_job_zone=target_job_zone,
            bridge_skills=bridge_skills,
            majors=focused_majors,
            activities=suggested_activities,
        )
        explanation = self._readiness_explanation(
            career=career,
            current_job_zone=current_job_zone,
            current_zone_label=current_zone_label,
            target_job_zone=target_job_zone,
            target_zone_label=target_zone_label,
            readiness_gap=readiness_gap,
            bridge_skills=bridge_skills,
            majors=focused_majors,
            target_reference=target_reference,
        )
        return ReadinessPlan(
            current_job_zone=current_job_zone,
            current_zone_label=current_zone_label,
            target_job_zone=target_job_zone,
            target_zone_label=target_zone_label,
            readiness_gap=readiness_gap,
            status=status,
            target_reference=target_reference,
            bridge_skills=bridge_skills,
            recommended_majors=focused_majors,
            suggested_activities=suggested_activities[:3],
            milestone_path=milestone_path,
            explanation=explanation,
        )

    def _readiness_major_names(
        self,
        state: StudentState,
        career: Entity,
        bridge_analysis: BridgeAnalysis,
        *,
        limit: int = 3,
    ) -> list[str]:
        career_vector = self.occupation_skill_vectors[career.id]
        missing_vector = self._missing_skill_vector(state.combined_skills, career_vector)
        bridge_vector = {
            evidence.target_skill_id: max(evidence.reachability, evidence.efficiency)
            for evidence in bridge_analysis.ranked_paths(limit=8)
            if evidence.reachability >= 0.1
        }
        scored: list[tuple[str, float]] = []
        for major in self.majors:
            major_vector = entity_skill_vector(self.graph, major.id, (RelationType.DEVELOPS_SKILL,))
            gap_support = weighted_overlap(major_vector, missing_vector)
            bridge_support = weighted_overlap(major_vector, bridge_vector) if bridge_vector else 0.0
            path_bonus = 0.0
            for relation in self.graph.relations_from(major.id, RelationType.LEADS_TO):
                if relation.object_id == career.id:
                    path_bonus = max(path_bonus, relation.weight)
                    continue
                for specialization in self.graph.relations_from(relation.object_id, RelationType.SPECIALIZES_TO):
                    if specialization.object_id == career.id:
                        path_bonus = max(path_bonus, relation.weight * specialization.weight)
            interest_bonus = 0.15 if any(match.id == major.id for match in state.matched_major_interests) else 0.0
            score = clamp(
                (gap_support * 0.55)
                + (bridge_support * 0.20)
                + (path_bonus * 0.20)
                + (interest_bonus * 0.05)
            )
            if score >= 0.1:
                scored.append((major.name, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return [name for name, _ in scored[:limit]]

    def _job_zone_reference_lines(self, career: Entity) -> list[str]:
        reference = career.attributes.get("job_zone_reference") or {}
        if not isinstance(reference, dict):
            return []
        lines: list[str] = []
        if reference.get("name"):
            lines.append(f"O*NET 준비도: {reference['name']}")
        if reference.get("education"):
            lines.append(f"교육 기준: {reference['education']}")
        if reference.get("experience"):
            lines.append(f"경험 기준: {reference['experience']}")
        if reference.get("job_training"):
            lines.append(f"훈련 기준: {reference['job_training']}")
        return lines[:4]

    def _readiness_status(
        self,
        current_job_zone: int,
        target_job_zone: int | None,
        majors: list[str],
    ) -> str:
        if target_job_zone is None:
            return "준비도 정보가 부족해 개별 경로 해석이 우선입니다."
        gap = target_job_zone - current_job_zone
        if gap <= 0:
            return "현재 준비도 범위 안에서 직접 탐색 가능한 진로입니다."
        if gap == 1:
            return "한 단계 성장형 경로가 필요한 진로입니다."
        if majors:
            return "전공을 경유하는 성장 경로가 필요한 진로입니다."
        return "브리지 역량을 누적하며 단계적으로 접근해야 하는 진로입니다."

    def _readiness_milestones(
        self,
        *,
        career: Entity,
        current_job_zone: int,
        target_job_zone: int | None,
        bridge_skills: list[str],
        majors: list[str],
        activities: list[str],
    ) -> list[str]:
        milestones = [f"현재 추정 준비도: Job Zone {current_job_zone} ({job_zone_label(current_job_zone)})"]
        if isinstance(target_job_zone, int) and target_job_zone > current_job_zone:
            if bridge_skills:
                milestones.append(f"핵심 브리지 역량 확보: {', '.join(bridge_skills[:2])}")
            if majors:
                milestones.append(f"경유 전공 축: {', '.join(majors[:2])}")
            if activities:
                milestones.append(f"보조 활동 축: {', '.join(activities[:2])}")
            milestones.append(f"도달 목표: Job Zone {target_job_zone} 직업 {career.name}")
            return milestones

        milestones.append(f"현재 구조 안에서 {career.name} 탐색을 시작할 수 있습니다")
        if bridge_skills:
            milestones.append(f"초기 보완 역량: {', '.join(bridge_skills[:2])}")
        return milestones

    def _readiness_explanation(
        self,
        *,
        career: Entity,
        current_job_zone: int,
        current_zone_label: str,
        target_job_zone: int | None,
        target_zone_label: str | None,
        readiness_gap: int | None,
        bridge_skills: list[str],
        majors: list[str],
        target_reference: list[str],
    ) -> str:
        parts = [
            f"현재 학생 상태는 Job Zone {current_job_zone}({current_zone_label})에 가깝습니다"
        ]
        if isinstance(target_job_zone, int):
            parts.append(
                f"{career.name}는 Job Zone {target_job_zone}({target_zone_label}) 진로로 해석됩니다"
            )
        if readiness_gap is not None and readiness_gap > 0:
            if majors:
                parts.append(f"{', '.join(majors[:2])} 전공 축을 거치면 구조적으로 유리합니다")
            if bridge_skills:
                parts.append(f"핵심 브리지 역량은 {', '.join(bridge_skills[:3])} 입니다")
        elif bridge_skills:
            parts.append(f"{', '.join(bridge_skills[:2])} 역량을 보완하면 진입 경로가 더 안정됩니다")
        if target_reference:
            parts.append(target_reference[0])
        return ". ".join(parts) + "."

    def _skill_level_map(self, profile: StudentProfile) -> dict[str, float]:
        levels: dict[str, float] = {}
        for skill_input in profile.skills:
            levels[self._skill_level_key(skill_input.name)] = skill_input.level
        return levels

    def _skill_level_key(self, value: str) -> str:
        return " ".join(value.split()).strip().casefold()

    def _resolve_simulated_activity(self, activity_name: str) -> tuple[Entity, str]:
        direct_matches = [
            (entity, score)
            for entity, score in self.graph.search(EntityType.ACTIVITY, activity_name, limit=5)
            if entity.attributes.get("activity_kind") == "student"
        ]
        if direct_matches and direct_matches[0][1] >= 0.95:
            return (
                direct_matches[0][0],
                f"'{activity_name}' 입력이 학생 활동 엔티티 '{direct_matches[0][0].name}'와 직접 일치했습니다.",
            )

        interpretation = self.profile_interpreter.interpret_profile({"activities": [activity_name]})
        for resolution in interpretation.entities:
            if resolution.source_field != "activities":
                continue
            entity = self.graph.entity(resolution.entity_id)
            if entity.type == EntityType.ACTIVITY and entity.attributes.get("activity_kind") == "student":
                return entity, resolution.reasoning

        if direct_matches:
            return (
                direct_matches[0][0],
                f"'{activity_name}' 입력을 가장 가까운 학생 활동 엔티티 '{direct_matches[0][0].name}'로 연결했습니다.",
            )

        raise ValueError("학생 활동 노드로 해석할 수 있는 활동을 찾지 못했습니다.")

    def _append_unique_entity(self, entities: list[Entity], entity: Entity) -> None:
        if any(existing.id == entity.id for existing in entities):
            return
        entities.append(entity)

    def _register_activity(
        self,
        state: StudentState,
        activity: Entity,
        *,
        strength: float,
        source_label: str,
    ) -> None:
        self._append_unique_entity(state.matched_activities, activity)
        for relation in self.graph.relations_from(activity.id, RelationType.DEVELOPS_SKILL):
            inferred = clamp(relation.weight * 0.7 * strength)
            state.inferred_skills[relation.object_id] = max(
                state.inferred_skills.get(relation.object_id, 0.0),
                inferred,
            )
            state.skill_sources[relation.object_id].append(source_label)

    def _expand_skill_map(self, merged: dict[str, float], state: StudentState) -> dict[str, float]:
        expanded = dict(merged)
        for skill_id, score in list(merged.items()):
            self._propagate_prerequisites(
                skill_id=skill_id,
                weight=score * 0.75,
                expanded=expanded,
                state=state,
                remaining_depth=2,
                source_name=self.graph.entity(skill_id).name,
            )
        return expanded

    def _propagate_prerequisites(
        self,
        *,
        skill_id: str,
        weight: float,
        expanded: dict[str, float],
        state: StudentState,
        remaining_depth: int,
        source_name: str,
    ) -> None:
        if remaining_depth <= 0 or weight <= 0:
            return
        for relation in self.graph.relations_from(skill_id, RelationType.HAS_PREREQUISITE):
            propagated = clamp(weight * relation.weight)
            if propagated <= 0:
                continue
            expanded[relation.object_id] = max(expanded.get(relation.object_id, 0.0), propagated)
            state.skill_sources[relation.object_id].append(f"전제역량:{source_name}")
            self._propagate_prerequisites(
                skill_id=relation.object_id,
                weight=propagated * 0.7,
                expanded=expanded,
                state=state,
                remaining_depth=remaining_depth - 1,
                source_name=source_name,
            )

    def _skill_signals(
        self, skill_map: dict[str, float], state: StudentState, *, declared: bool
    ) -> list[SkillSignal]:
        signals: list[SkillSignal] = []
        for skill_id, score in sorted(skill_map.items(), key=lambda item: item[1], reverse=True)[:10]:
            signals.append(
                SkillSignal(
                    skill_name=self.graph.entity(skill_id).name,
                    score=round(score, 4),
                    source="직접 입력" if declared else ", ".join(state.skill_sources[skill_id]),
                )
            )
        return signals

    def _trait_alignment(self, state: StudentState, career: Entity) -> float:
        if not state.matched_traits:
            return 0.0
        trait_weights = {
            relation.object_id: relation.weight
            for relation in self.graph.relations_from(career.id, RelationType.ALIGNED_WITH_TRAIT)
        }
        if not trait_weights:
            return 0.0
        matched_scores = [trait_weights.get(trait.id, 0.0) for trait in state.matched_traits]
        return sum(matched_scores) / len(matched_scores)

    def _career_interest_alignment(self, state: StudentState, career: Entity) -> float:
        if not state.matched_career_interests:
            return 0.0
        scores = [soft_match_score(interest.name, career.name) for interest in state.matched_career_interests]
        for relation in self.graph.relations_to(career.id, RelationType.SPECIALIZES_TO):
            source = self.graph.entity(relation.subject_id)
            if any(interest.id == source.id for interest in state.matched_career_interests):
                scores.append(relation.weight)
        return max(scores, default=0.0)

    def _market_alignment(self, student_skills: dict[str, float], career: Entity) -> float:
        market_scores: list[float] = []
        for relation in self.graph.relations_to(career.id, RelationType.SPECIALIZES_TO):
            source = self.graph.entity(relation.subject_id)
            if source.attributes.get("profile_kind") != "market_sector":
                continue
            market_vector = entity_skill_vector(self.graph, source.id, (RelationType.REQUIRES_SKILL,))
            market_scores.append(weighted_overlap(student_skills, market_vector) * relation.weight)
        return max(market_scores, default=0.0)

    def _activity_alignment(self, state: StudentState, career: Entity) -> float:
        if not state.matched_activities:
            return 0.0
        scores: list[float] = []
        for activity in state.matched_activities:
            for relation in self.graph.relations_from(activity.id, RelationType.SUPPORTS_CAREER):
                if relation.object_id == career.id:
                    scores.append(relation.weight)
                    continue
                for specialization in self.graph.relations_from(relation.object_id, RelationType.SPECIALIZES_TO):
                    if specialization.object_id == career.id:
                        scores.append(relation.weight * specialization.weight)
        return max(scores, default=0.0)

    def _major_alignment(self, state: StudentState, career: Entity) -> float:
        if not state.matched_major_interests:
            return 0.0
        career_vector = self.occupation_skill_vectors[career.id]
        scores: list[float] = []
        for major in state.matched_major_interests:
            major_vector = entity_skill_vector(self.graph, major.id, (RelationType.DEVELOPS_SKILL,))
            scores.append(weighted_overlap(major_vector, career_vector) * 0.7)
            for relation in self.graph.relations_from(major.id, RelationType.LEADS_TO):
                if relation.object_id == career.id:
                    scores.append(relation.weight)
                    continue
                for specialization in self.graph.relations_from(relation.object_id, RelationType.SPECIALIZES_TO):
                    if specialization.object_id == career.id:
                        scores.append(relation.weight * specialization.weight)
        return max(scores, default=0.0)

    def _zone_fit(self, profile: StudentProfile, career: Entity) -> float:
        if not profile.preferred_job_zones:
            return 1.0
        job_zone = career.attributes.get("job_zone")
        return 1.0 if job_zone in profile.preferred_job_zones else 0.4

    def _top_matched_skills(
        self, student_skills: dict[str, float], career_vector: dict[str, float], limit: int = 5
    ) -> list[str]:
        matches = [
            (skill_id, min(student_skills.get(skill_id, 0.0), weight))
            for skill_id, weight in career_vector.items()
            if student_skills.get(skill_id, 0.0) > 0.0
        ]
        matches.sort(key=lambda item: item[1], reverse=True)
        return [self.graph.entity(skill_id).name for skill_id, _ in matches[:limit]]

    def _top_missing_skills(
        self, student_skills: dict[str, float], career_vector: dict[str, float], limit: int = 5
    ) -> list[str]:
        gaps: list[tuple[str, float]] = []
        for skill_id, weight in career_vector.items():
            gap = max(weight - student_skills.get(skill_id, 0.0), 0.0)
            if gap > 0.1:
                gaps.append((skill_id, gap))
        gaps.sort(key=lambda item: item[1], reverse=True)
        return [self.graph.entity(skill_id).name for skill_id, _ in gaps[:limit]]

    def _top_knowledge_areas(self, career: Entity, limit: int = 3) -> list[str]:
        relations = sorted(
            self.graph.relations_from(career.id, RelationType.REQUIRES_KNOWLEDGE),
            key=lambda relation: relation.weight,
            reverse=True,
        )
        return [self.graph.entity(relation.object_id).name for relation in relations[:limit]]

    def _matched_trait_names(self, state: StudentState, career: Entity) -> list[str]:
        trait_ids = {
            relation.object_id
            for relation in self.graph.relations_from(career.id, RelationType.ALIGNED_WITH_TRAIT)
        }
        return [trait.name for trait in state.matched_traits if trait.id in trait_ids][:3]

    def _career_contexts(self, career: Entity, limit: int = 3) -> list[str]:
        contexts = [
            (self.graph.entity(relation.subject_id).name, relation.weight)
            for relation in self.graph.relations_to(career.id, RelationType.SPECIALIZES_TO)
        ]
        contexts.sort(key=lambda item: item[1], reverse=True)
        return [name for name, _ in contexts[:limit]]

    def _supporting_major_names(
        self, state: StudentState, career: Entity, limit: int = 3
    ) -> list[str]:
        career_vector = self.occupation_skill_vectors[career.id]
        scored: list[tuple[str, float]] = []
        for major in self.majors:
            major_vector = entity_skill_vector(self.graph, major.id, (RelationType.DEVELOPS_SKILL,))
            score = weighted_overlap(major_vector, career_vector)
            if any(match.id == major.id for match in state.matched_major_interests):
                score = clamp(score + 0.15)
            if score >= 0.1:
                scored.append((major.name, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return [name for name, _ in scored[:limit]]

    def _supporting_activity_names(
        self, state: StudentState, career: Entity, limit: int = 3
    ) -> list[str]:
        target_vector = self._missing_skill_vector(state.combined_skills, self.occupation_skill_vectors[career.id])
        scored: list[tuple[str, float]] = []
        for activity in self.student_activities:
            activity_vector = entity_skill_vector(self.graph, activity.id, (RelationType.DEVELOPS_SKILL,))
            score = weighted_overlap(activity_vector, target_vector)
            for relation in self.graph.relations_from(activity.id, RelationType.SUPPORTS_CAREER):
                if relation.object_id == career.id:
                    score = clamp(score + (0.2 * relation.weight))
            if score >= 0.1:
                scored.append((activity.name, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return [name for name, _ in scored[:limit]]

    def _major_key_skills(self, major: Entity, limit: int = 4) -> list[str]:
        relations = sorted(
            self.graph.relations_from(major.id, RelationType.DEVELOPS_SKILL),
            key=lambda relation: relation.weight,
            reverse=True,
        )
        return [self.graph.entity(relation.object_id).name for relation in relations[:limit]]

    def _major_connected_careers(
        self, major: Entity, top_careers: list[CareerRecommendation], limit: int = 3
    ) -> list[str]:
        top_career_ids = {career.career_id for career in top_careers}
        relations = [
            relation
            for relation in self.graph.relations_from(major.id, RelationType.LEADS_TO)
            if relation.object_id in top_career_ids
        ]
        relations.sort(key=lambda relation: relation.weight, reverse=True)
        return [self.graph.entity(relation.object_id).name for relation in relations[:limit]]

    def _activity_focus_skills(
        self, activity: Entity, missing_vector: dict[str, float], limit: int = 4
    ) -> list[str]:
        activity_vector = entity_skill_vector(self.graph, activity.id, (RelationType.DEVELOPS_SKILL,))
        ranked = sorted(
            activity_vector.items(),
            key=lambda item: min(item[1], missing_vector.get(item[0], 0.0)),
            reverse=True,
        )
        focus = [self.graph.entity(skill_id).name for skill_id, _ in ranked if missing_vector.get(skill_id, 0.0) > 0]
        if focus:
            return focus[:limit]
        return [self.graph.entity(skill_id).name for skill_id, _ in ranked[:limit]]

    def _activity_support_targets(
        self,
        activity: Entity,
        top_careers: list[CareerRecommendation],
        top_majors: list[MajorRecommendation],
        limit: int = 4,
    ) -> list[str]:
        top_career_ids = {career.career_id for career in top_careers}
        top_major_ids = {major.major_id for major in top_majors}
        targets: list[tuple[str, float]] = []
        for relation in self.graph.relations_from(activity.id, RelationType.SUPPORTS_CAREER):
            if relation.object_id in top_career_ids:
                targets.append((self.graph.entity(relation.object_id).name, relation.weight))
        for relation in self.graph.relations_from(activity.id, RelationType.SUPPORTS_MAJOR):
            if relation.object_id in top_major_ids:
                targets.append((self.graph.entity(relation.object_id).name, relation.weight))
        targets.sort(key=lambda item: item[1], reverse=True)

        ordered: list[str] = []
        seen: set[str] = set()
        for name, _ in targets:
            if name in seen:
                continue
            seen.add(name)
            ordered.append(name)
        return ordered[:limit]

    def _build_paths(
        self,
        *,
        career: Entity,
        matched_skills: list[str],
        majors: list[str],
        activities: list[str],
        bridge_analysis: BridgeAnalysis,
    ) -> list[str]:
        paths: list[str] = []
        for skill_name in matched_skills[:2]:
            paths.append(f"학생 -> 보유 역량 -> {skill_name} -> 진로 요구 역량 -> {career.name}")
        for bridge_text in self._bridge_path_texts(bridge_analysis, career, limit=2):
            paths.append(bridge_text)
        for major_name in majors[:1]:
            paths.append(f"{major_name} -> 역량 형성 -> 진로 연결 -> {career.name}")
        for activity_name in activities[:1]:
            paths.append(f"학생 -> 활동 경험 -> {activity_name} -> 역량 강화 -> {career.name}")
        return paths

    def _build_next_actions(
        self,
        *,
        market_signal: MarketSignalSummary,
        readiness_plan: ReadinessPlan,
        bridge_skills: list[str],
        missing_skills: list[str],
        suggested_activities: list[str],
        recommended_knowledge: list[str],
        supporting_majors: list[str],
    ) -> list[str]:
        actions: list[str] = []
        if readiness_plan.readiness_gap is not None and readiness_plan.target_job_zone is not None:
            if readiness_plan.readiness_gap > 0:
                actions.append(
                    f"준비도 경로: Job Zone {readiness_plan.current_job_zone} -> {readiness_plan.target_job_zone}"
                )
            else:
                actions.append(
                    f"현재 준비도 범위: Job Zone {readiness_plan.current_job_zone} 안에서 탐색 가능"
                )
        if readiness_plan.recommended_majors:
            actions.append(f"경유 전공: {', '.join(readiness_plan.recommended_majors[:2])}")
        if bridge_skills:
            actions.append(f"현재 역량으로 이어갈 브리지 목표: {', '.join(bridge_skills[:2])}")
        if market_signal.bridge_signal_skills:
            actions.append(f"시장 신호 기준 보완 역량: {', '.join(market_signal.bridge_signal_skills[:2])}")
        elif market_signal.technology_highlights:
            actions.append(f"시장 기술 단서: {', '.join(market_signal.technology_highlights[:1])}")
        if missing_skills:
            actions.append(f"우선 보완할 역량: {', '.join(missing_skills[:2])}")
        if suggested_activities:
            actions.append(f"권장 활동: {', '.join(suggested_activities[:2])}")
        if supporting_majors:
            actions.append(f"검토할 전공 축: {', '.join(supporting_majors[:2])}")
        if recommended_knowledge:
            actions.append(f"학습할 지식 영역: {', '.join(recommended_knowledge[:2])}")
        return actions[:4]

    def _career_explanation(
        self,
        *,
        market_signal: MarketSignalSummary,
        readiness_plan: ReadinessPlan,
        career: Entity,
        matched_skills: list[str],
        bridge_skills: list[str],
        matched_traits: list[str],
        contexts: list[str],
    ) -> str:
        parts: list[str] = []
        if readiness_plan.readiness_gap is not None and readiness_plan.target_job_zone is not None:
            if readiness_plan.readiness_gap > 0:
                parts.append(
                    f"현재 준비도(Job Zone {readiness_plan.current_job_zone})에서 "
                    f"Job Zone {readiness_plan.target_job_zone}로 올라가는 경로가 보입니다"
                )
            else:
                parts.append(
                    f"현재 준비도(Job Zone {readiness_plan.current_job_zone}) 안에서 탐색 가능한 진로입니다"
                )
        if matched_skills:
            parts.append(f"{', '.join(matched_skills[:2])} 역량이 직접 맞물립니다")
        if bridge_skills:
            parts.append(f"{', '.join(bridge_skills[:2])} 역량은 현재 상태에서 브리지 경로로 도달 가능합니다")
        if market_signal.market_contexts:
            parts.append(f"{', '.join(market_signal.market_contexts[:2])} 시장 맥락과 연결됩니다")
        if market_signal.bridge_signal_skills:
            parts.append(
                f"시장 기술 축에서는 {', '.join(market_signal.bridge_signal_skills[:2])} 보완이 중요합니다"
            )
        if contexts:
            parts.append(f"{', '.join(contexts[:2])} 맥락과 연결됩니다")
        if matched_traits:
            parts.append(f"{', '.join(matched_traits[:2])} 성향이 이 진로와 잘 맞습니다")
        if not parts:
            parts.append(f"{career.name}와의 구조적 연결이 확인되었습니다")
        return ". ".join(parts) + "."

    def _student_summary(
        self,
        profile: StudentProfile,
        state: StudentState,
        top_careers: list[CareerRecommendation],
        readiness: StudentReadinessState,
    ) -> str:
        strong_skills = [
            self.graph.entity(skill_id).name
            for skill_id, _ in sorted(state.combined_skills.items(), key=lambda item: item[1], reverse=True)[:3]
        ]
        top_career_name = top_careers[0].career_name if top_careers else "추가 해석 필요"
        interests = ", ".join(entity.name for entity in state.matched_career_interests[:2])
        traits = ", ".join(entity.name for entity in state.matched_traits[:2])

        parts: list[str] = []
        if strong_skills:
            parts.append(f"현재 두드러진 의미 축은 {', '.join(strong_skills)} 입니다")
        if interests:
            parts.append(f"관심 신호는 {interests} 방향으로 모입니다")
        if traits:
            parts.append(f"성향 단서는 {traits} 입니다")
        parts.append(
            f"운영상 현재 준비도는 Job Zone {readiness.current_job_zone}"
            f"({job_zone_label(readiness.current_job_zone)})에 가깝습니다"
        )
        if readiness.next_job_zone is not None and readiness.priority_bridge_skills:
            parts.append(
                f"다음 단계인 Job Zone {readiness.next_job_zone}로 가려면 "
                f"{', '.join(readiness.priority_bridge_skills[:2])} 브리지 역량을 우선 강화하는 편이 좋습니다"
            )
        parts.append(f"현재 구조상 가장 강하게 연결되는 대표 진로는 {top_career_name} 입니다")
        return ". ".join(parts) + "."

    def _aggregate_target_vector(self, career_ids: list[str]) -> dict[str, float]:
        aggregate: dict[str, float] = defaultdict(float)
        for career_id in career_ids:
            for skill_id, weight in self.occupation_skill_vectors.get(career_id, {}).items():
                aggregate[skill_id] = max(aggregate[skill_id], weight)
        return dict(aggregate)

    def _missing_skill_vector(
        self, student_skills: dict[str, float], target_vector: dict[str, float]
    ) -> dict[str, float]:
        gaps: dict[str, float] = {}
        for skill_id, weight in target_vector.items():
            gap = max(weight - student_skills.get(skill_id, 0.0), 0.0)
            if gap > 0.05:
                gaps[skill_id] = gap
        return gaps

    def _career_impacts(
        self,
        *,
        activity: Entity,
        baseline_report: RecommendationReport,
        simulated_report: RecommendationReport,
    ) -> list[CareerImpact]:
        baseline_index = {
            recommendation.career_id: (index + 1, recommendation)
            for index, recommendation in enumerate(baseline_report.top_careers)
        }
        simulated_index = {
            recommendation.career_id: (index + 1, recommendation)
            for index, recommendation in enumerate(simulated_report.top_careers)
        }

        impacts: list[CareerImpact] = []
        for career_id in { *baseline_index.keys(), *simulated_index.keys() }:
            baseline_rank, baseline = baseline_index.get(career_id, (None, None))
            simulated_rank, simulated = simulated_index.get(career_id, (None, None))
            career_name = (
                simulated.career_name
                if simulated is not None
                else baseline.career_name
                if baseline is not None
                else self.graph.entity(career_id).name
            )
            baseline_score = baseline.score if baseline is not None else 0.0
            simulated_score = simulated.score if simulated is not None else 0.0
            score_delta = round(simulated_score - baseline_score, 4)
            if abs(score_delta) < 0.0001:
                continue

            rank_change: int | None = None
            if baseline_rank is not None and simulated_rank is not None:
                rank_change = baseline_rank - simulated_rank

            impacts.append(
                CareerImpact(
                    career_id=career_id,
                    career_name=career_name,
                    baseline_score=round(baseline_score, 4),
                    simulated_score=round(simulated_score, 4),
                    score_delta=score_delta,
                    score_delta_percent=round(score_delta * 100.0, 2),
                    baseline_rank=baseline_rank,
                    simulated_rank=simulated_rank,
                    rank_change=rank_change,
                    explanation=self._career_impact_explanation(activity.name, baseline, simulated, score_delta, rank_change),
                )
            )

        impacts.sort(key=lambda item: (item.score_delta, -(item.simulated_rank or 999), item.career_name), reverse=True)
        return impacts

    def _career_impact_explanation(
        self,
        activity_name: str,
        baseline: CareerRecommendation | None,
        simulated: CareerRecommendation | None,
        score_delta: float,
        rank_change: int | None,
    ) -> str:
        parts: list[str] = []
        if score_delta > 0:
            parts.append(f"{activity_name} 활동 주입으로 적합도가 {score_delta * 100:.1f}%p 상승했습니다")
        else:
            parts.append(f"{activity_name} 활동 주입 후 상대 점수가 {abs(score_delta) * 100:.1f}%p 하락했습니다")

        if baseline is not None and simulated is not None:
            new_matches = [
                skill for skill in simulated.matched_skills if skill not in baseline.matched_skills
            ]
            reduced_gaps = [
                skill for skill in baseline.missing_skills if skill not in simulated.missing_skills
            ]
            if new_matches:
                parts.append(f"{', '.join(new_matches[:2])} 역량이 더 직접적으로 연결됩니다")
            if reduced_gaps:
                parts.append(f"{', '.join(reduced_gaps[:2])} 역량 격차가 줄어듭니다")
        elif simulated is not None:
            parts.append("이 활동을 넣었을 때 상위 추천 목록 안으로 새롭게 들어옵니다")

        if rank_change is not None and rank_change > 0:
            parts.append(f"추천 순위가 {rank_change}단계 상승합니다")

        return ". ".join(parts) + "."

    def _skill_impacts(
        self,
        *,
        activity: Entity,
        baseline_state: StudentState,
        simulated_state: StudentState,
    ) -> list[SkillImpact]:
        direct_activity_skills = {
            relation.object_id
            for relation in self.graph.relations_from(activity.id, RelationType.DEVELOPS_SKILL)
        }

        impacts: list[SkillImpact] = []
        for skill_id in set(baseline_state.combined_skills) | set(simulated_state.combined_skills):
            baseline_score = baseline_state.combined_skills.get(skill_id, 0.0)
            simulated_score = simulated_state.combined_skills.get(skill_id, 0.0)
            score_delta = round(simulated_score - baseline_score, 4)
            if score_delta < 0.04:
                continue

            if skill_id in direct_activity_skills:
                explanation = f"{activity.name} 활동이 이 역량을 직접 강화합니다."
            else:
                explanation = (
                    f"{activity.name} 활동이 강화한 역량이 전제 관계를 따라 전파되어 "
                    f"{self.graph.entity(skill_id).name}까지 영향을 줍니다."
                )

            impacts.append(
                SkillImpact(
                    skill_name=self.graph.entity(skill_id).name,
                    baseline_score=round(baseline_score, 4),
                    simulated_score=round(simulated_score, 4),
                    score_delta=score_delta,
                    explanation=explanation,
                )
            )

        impacts.sort(key=lambda item: (item.score_delta, item.simulated_score), reverse=True)
        return impacts

    def _action_summary(
        self,
        activity_name: str,
        already_present: bool,
        career_impacts: list[CareerImpact],
        skill_impacts: list[SkillImpact],
    ) -> str:
        if already_present:
            return (
                f"'{activity_name}' 활동은 이미 현재 학생 상태에 반영되어 있어 "
                "예상 변화가 크지 않습니다."
            )

        top_career = next((impact for impact in career_impacts if impact.score_delta > 0), None)
        top_skill = next((impact for impact in skill_impacts if impact.score_delta > 0), None)
        if top_career is not None and top_skill is not None:
            return (
                f"이 활동을 완료하면 '{top_career.career_name}' 적합도가 "
                f"{top_career.score_delta_percent:.1f}%p 상승하고, "
                f"'{top_skill.skill_name}' 역량이 보완됩니다."
            )
        if top_career is not None:
            return (
                f"이 활동을 완료하면 '{top_career.career_name}' 적합도가 "
                f"{top_career.score_delta_percent:.1f}%p 상승합니다."
            )
        if top_skill is not None:
            return f"이 활동은 '{top_skill.skill_name}' 역량을 중심으로 학생 상태를 강화합니다."
        return "이 활동은 현재 추천 구조에 큰 변화를 만들지 않지만, 의미 구조상 탐색 가치가 있습니다."

    def _future_feedback(
        self,
        activity_name: str,
        already_present: bool,
        career_impacts: list[CareerImpact],
        skill_impacts: list[SkillImpact],
    ) -> list[str]:
        feedback: list[str] = []
        if already_present:
            feedback.append("이미 반영된 활동이라 추가 변화보다 현재 구조 재해석에 가깝습니다.")
        for impact in career_impacts[:3]:
            direction = "상승" if impact.score_delta >= 0 else "하락"
            feedback.append(
                f"{impact.career_name}: 적합도 {abs(impact.score_delta_percent):.1f}%p {direction}"
            )
        for impact in skill_impacts[:3]:
            feedback.append(
                f"{impact.skill_name}: 역량 점수 {impact.score_delta * 100:.1f}%p 보완"
            )
        if not feedback:
            feedback.append(f"{activity_name} 활동은 현재 구조상 변화폭이 제한적입니다.")
        return feedback

    def _bridge_skill_names(self, bridge_analysis: BridgeAnalysis, limit: int = 3) -> list[str]:
        names: list[str] = []
        for evidence in bridge_analysis.ranked_paths(limit=limit * 2):
            if evidence.bridge_count <= 0 or evidence.reachability < 0.12:
                continue
            names.append(self.graph.entity(evidence.target_skill_id).name)
            if len(names) >= limit:
                break
        return names

    def _bridge_path_texts(
        self,
        bridge_analysis: BridgeAnalysis,
        career: Entity,
        *,
        limit: int = 2,
    ) -> list[str]:
        texts: list[str] = []
        for evidence in bridge_analysis.ranked_paths(limit=limit * 3):
            if evidence.bridge_count <= 0 or evidence.reachability < 0.12:
                continue
            skill_names = [
                self.graph.entity(node_id).name
                for node_id in evidence.path_node_ids
                if node_id != STUDENT_SOURCE_ID
            ]
            if len(skill_names) < 2:
                continue
            text = f"학생 -> 보유 역량 -> {skill_names[0]}"
            for bridge_name in skill_names[1:-1]:
                text += f" -> 브리지 역량 -> {bridge_name}"
            text += f" -> 진로 요구 역량 -> {skill_names[-1]} -> {career.name}"
            texts.append(text)
            if len(texts) >= limit:
                break
        return texts

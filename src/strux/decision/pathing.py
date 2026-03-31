from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx

from strux.interpretation.normalization import clamp
from strux.ontology.graph import OntologyGraph
from strux.ontology.model import EntityType, RelationType


STUDENT_SOURCE_ID = "__student_state__"


@dataclass
class SkillPathEvidence:
    target_skill_id: str
    total_cost: float = 0.0
    reachability: float = 0.0
    efficiency: float = 0.0
    frontier_signal: float = 0.0
    path_strength: float = 0.0
    path_node_ids: list[str] = field(default_factory=list)

    @property
    def bridge_count(self) -> int:
        return max(len(self.path_node_ids) - 2, 0)

    @property
    def bridge_skill_ids(self) -> list[str]:
        if len(self.path_node_ids) <= 2:
            return []
        return self.path_node_ids[1:-1]


@dataclass
class BridgeAnalysis:
    bridge_reachability: float
    path_efficiency: float
    frontier_signal: float
    skill_paths: dict[str, SkillPathEvidence] = field(default_factory=dict)

    def ranked_paths(self, limit: int = 5) -> list[SkillPathEvidence]:
        ranked = sorted(
            self.skill_paths.values(),
            key=lambda item: (item.reachability, item.frontier_signal, item.efficiency),
            reverse=True,
        )
        return ranked[:limit]


class BridgePathPlanner:
    def __init__(self, graph: OntologyGraph) -> None:
        self.graph = graph
        self.skill_graph = self._build_skill_graph()

    def analyze(
        self,
        student_skills: dict[str, float],
        target_skills: dict[str, float],
    ) -> BridgeAnalysis:
        if not target_skills:
            return BridgeAnalysis(
                bridge_reachability=1.0,
                path_efficiency=1.0,
                frontier_signal=1.0,
            )

        source_skills = {
            skill_id: clamp(score)
            for skill_id, score in student_skills.items()
            if score >= 0.05 and self.graph.maybe_entity(skill_id) is not None
        }
        if not source_skills:
            return BridgeAnalysis(bridge_reachability=0.0, path_efficiency=0.0, frontier_signal=0.0)

        request_graph = self.skill_graph.copy()
        request_graph.add_node(STUDENT_SOURCE_ID)
        for skill_id, score in source_skills.items():
            if not request_graph.has_node(skill_id):
                continue
            request_graph.add_edge(
                STUDENT_SOURCE_ID,
                skill_id,
                cost=clamp(0.4 - (score * 0.25), lower=0.05, upper=0.4),
                strength=clamp(score),
                edge_kind="student-skill",
            )

        lengths, paths = nx.single_source_dijkstra(request_graph, STUDENT_SOURCE_ID, weight="cost")
        pagerank = self._personalized_pagerank(source_skills)
        max_pagerank = max(pagerank.values(), default=0.0) or 1.0

        total_target_weight = sum(target_skills.values()) or 1.0
        reachability_total = 0.0
        efficiency_total = 0.0
        frontier_total = 0.0
        skill_paths: dict[str, SkillPathEvidence] = {}

        for skill_id, target_weight in target_skills.items():
            evidence = self._path_evidence(
                skill_id=skill_id,
                source_skills=source_skills,
                request_graph=request_graph,
                lengths=lengths,
                paths=paths,
                pagerank=pagerank,
                max_pagerank=max_pagerank,
            )
            skill_paths[skill_id] = evidence
            reachability_total += target_weight * evidence.reachability
            efficiency_total += target_weight * evidence.efficiency
            frontier_total += target_weight * evidence.frontier_signal

        return BridgeAnalysis(
            bridge_reachability=clamp(reachability_total / total_target_weight),
            path_efficiency=clamp(efficiency_total / total_target_weight),
            frontier_signal=clamp(frontier_total / total_target_weight),
            skill_paths=skill_paths,
        )

    def _build_skill_graph(self) -> nx.DiGraph:
        graph = nx.DiGraph()
        for entity in self.graph.entities_by_type(EntityType.SKILL):
            graph.add_node(entity.id)

        for skill in self.graph.entities_by_type(EntityType.SKILL):
            for relation in self.graph.relations_from(skill.id, RelationType.HAS_PREREQUISITE):
                prerequisite_id = relation.object_id
                advanced_skill_id = skill.id
                if not graph.has_node(prerequisite_id) or not graph.has_node(advanced_skill_id):
                    continue
                graph.add_edge(
                    prerequisite_id,
                    advanced_skill_id,
                    cost=0.45 + ((1.0 - clamp(relation.weight)) * 1.1),
                    strength=clamp(relation.weight),
                    edge_kind="skill-bridge",
                )
        return graph

    def _personalized_pagerank(self, source_skills: dict[str, float]) -> dict[str, float]:
        if not source_skills:
            return {}
        total = sum(source_skills.values())
        personalization = {
            node_id: (source_skills.get(node_id, 0.0) / total)
            for node_id in self.skill_graph.nodes
        }
        return self._power_iteration_rank(personalization)

    def _power_iteration_rank(
        self,
        personalization: dict[str, float],
        *,
        alpha: float = 0.85,
        max_iter: int = 100,
        tol: float = 1.0e-6,
    ) -> dict[str, float]:
        nodes = list(self.skill_graph.nodes)
        if not nodes:
            return {}

        ranks = dict(personalization)
        for node_id in nodes:
            ranks.setdefault(node_id, 0.0)

        outgoing_strength = {
            node_id: sum(
                edge_data.get("strength", 1.0)
                for _, _, edge_data in self.skill_graph.out_edges(node_id, data=True)
            )
            for node_id in nodes
        }

        for _ in range(max_iter):
            next_ranks = {
                node_id: (1.0 - alpha) * personalization.get(node_id, 0.0)
                for node_id in nodes
            }
            dangling_mass = sum(
                ranks[node_id]
                for node_id in nodes
                if outgoing_strength[node_id] == 0.0
            )
            if dangling_mass:
                for node_id in nodes:
                    next_ranks[node_id] += alpha * dangling_mass * personalization.get(node_id, 0.0)

            for node_id in nodes:
                total_strength = outgoing_strength[node_id]
                if total_strength == 0.0:
                    continue
                for _, target_id, edge_data in self.skill_graph.out_edges(node_id, data=True):
                    transition = edge_data.get("strength", 1.0) / total_strength
                    next_ranks[target_id] += alpha * ranks[node_id] * transition

            delta = sum(abs(next_ranks[node_id] - ranks[node_id]) for node_id in nodes)
            ranks = next_ranks
            if delta <= tol * len(nodes):
                break

        normalization = sum(ranks.values()) or 1.0
        return {node_id: value / normalization for node_id, value in ranks.items()}

    def _path_evidence(
        self,
        *,
        skill_id: str,
        source_skills: dict[str, float],
        request_graph: nx.DiGraph,
        lengths: dict[str, float],
        paths: dict[str, list[str]],
        pagerank: dict[str, float],
        max_pagerank: float,
    ) -> SkillPathEvidence:
        if skill_id in source_skills:
            direct_score = clamp(source_skills[skill_id])
            total_cost = request_graph[STUDENT_SOURCE_ID][skill_id]["cost"]
            efficiency = clamp(1.0 - (total_cost * 0.5))
            frontier_signal = clamp((pagerank.get(skill_id, 0.0) / max_pagerank) if pagerank else direct_score)
            reachability = clamp((direct_score * 0.75) + (efficiency * 0.25))
            return SkillPathEvidence(
                target_skill_id=skill_id,
                total_cost=round(total_cost, 4),
                reachability=round(reachability, 4),
                efficiency=round(efficiency, 4),
                frontier_signal=round(frontier_signal, 4),
                path_strength=round(direct_score, 4),
                path_node_ids=[STUDENT_SOURCE_ID, skill_id],
            )

        if skill_id not in lengths:
            return SkillPathEvidence(target_skill_id=skill_id)

        path_node_ids = paths[skill_id]
        total_cost = lengths[skill_id]
        efficiency = clamp(1.0 / (1.0 + total_cost))
        frontier_signal = clamp((pagerank.get(skill_id, 0.0) / max_pagerank) if pagerank else 0.0)
        path_strength = 1.0
        for source_id, target_id in zip(path_node_ids, path_node_ids[1:]):
            path_strength *= request_graph[source_id][target_id].get("strength", 1.0)
        bridge_decay = 0.92 ** max(len(path_node_ids) - 2, 0)
        reachability = clamp(((efficiency * 0.6) + (frontier_signal * 0.4)) * bridge_decay * max(0.65, path_strength))

        return SkillPathEvidence(
            target_skill_id=skill_id,
            total_cost=round(total_cost, 4),
            reachability=round(reachability, 4),
            efficiency=round(efficiency, 4),
            frontier_signal=round(frontier_signal, 4),
            path_strength=round(path_strength, 4),
            path_node_ids=path_node_ids,
        )

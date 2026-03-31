from collections import defaultdict
from functools import lru_cache

from strux.ontology.graph import OntologyGraph
from strux.ontology.model import RelationType


def entity_skill_vector(
    graph: OntologyGraph,
    entity_id: str,
    relation_types: tuple[RelationType, ...],
    *,
    include_prerequisites: bool = True,
    prerequisite_decay: float = 0.7,
    max_depth: int = 2,
) -> dict[str, float]:
    vector: dict[str, float] = defaultdict(float)
    for relation_type in relation_types:
        for relation in graph.relations_from(entity_id, relation_type):
            vector[relation.object_id] = max(vector[relation.object_id], relation.weight)
            if include_prerequisites and max_depth > 0:
                _propagate_prerequisites(
                    graph,
                    relation.object_id,
                    relation.weight * prerequisite_decay,
                    vector,
                    prerequisite_decay=prerequisite_decay,
                    remaining_depth=max_depth - 1,
                )
    return dict(vector)


@lru_cache(maxsize=4096)
def _cached_prerequisites(
    graph_id: int, skill_id: str, remaining_depth: int
) -> tuple[tuple[str, float], ...]:
    graph = _GRAPH_REGISTRY[graph_id]
    if remaining_depth < 0:
        return ()
    entries: list[tuple[str, float]] = []
    for relation in graph.relations_from(skill_id, RelationType.HAS_PREREQUISITE):
        entries.append((relation.object_id, relation.weight))
        if remaining_depth > 0:
            for nested_skill_id, nested_weight in _cached_prerequisites(
                graph_id, relation.object_id, remaining_depth - 1
            ):
                entries.append((nested_skill_id, relation.weight * nested_weight))
    return tuple(entries)


_GRAPH_REGISTRY: dict[int, OntologyGraph] = {}


def _propagate_prerequisites(
    graph: OntologyGraph,
    skill_id: str,
    propagated_weight: float,
    vector: dict[str, float],
    *,
    prerequisite_decay: float,
    remaining_depth: int,
) -> None:
    graph_id = id(graph)
    _GRAPH_REGISTRY[graph_id] = graph
    for prerequisite_id, relation_weight in _cached_prerequisites(graph_id, skill_id, remaining_depth):
        vector[prerequisite_id] = max(
            vector[prerequisite_id],
            propagated_weight * relation_weight,
        )

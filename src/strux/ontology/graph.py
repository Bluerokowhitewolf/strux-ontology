from collections import defaultdict
from typing import Iterable

from strux.interpretation.normalization import normalize_label, soft_match_score
from strux.ontology.model import Entity, EntityType, Relation, RelationType


class OntologyGraph:
    def __init__(self) -> None:
        self.entities: dict[str, Entity] = {}
        self.relations: dict[tuple[str, RelationType, str], Relation] = {}
        self._outgoing: dict[str, list[Relation]] = defaultdict(list)
        self._incoming: dict[str, list[Relation]] = defaultdict(list)
        self._name_index: dict[EntityType, dict[str, str]] = defaultdict(dict)

    def add_entity(self, entity: Entity) -> Entity:
        existing = self.entities.get(entity.id)
        if existing is None:
            self.entities[entity.id] = entity
            self._index_entity(entity)
            return entity

        merged_aliases = list(dict.fromkeys([*existing.aliases, *entity.aliases]))
        merged_attributes = {**existing.attributes, **entity.attributes}
        merged_provenance = [*existing.provenance, *entity.provenance]
        updated = existing.model_copy(
            update={
                "description": entity.description or existing.description,
                "aliases": merged_aliases,
                "attributes": merged_attributes,
                "provenance": merged_provenance,
            }
        )
        self.entities[entity.id] = updated
        self._index_entity(updated)
        return updated

    def add_relation(self, relation: Relation) -> Relation:
        key = (relation.subject_id, relation.predicate, relation.object_id)
        existing = self.relations.get(key)
        if existing is None:
            self.relations[key] = relation
            self._outgoing[relation.subject_id].append(relation)
            self._incoming[relation.object_id].append(relation)
            return relation

        updated = existing.model_copy(
            update={
                "weight": max(existing.weight, relation.weight),
                "attributes": {**existing.attributes, **relation.attributes},
                "evidence": [*existing.evidence, *relation.evidence],
            }
        )
        self.relations[key] = updated

        self._outgoing[relation.subject_id] = [
            updated if rel.object_id == relation.object_id and rel.predicate == relation.predicate else rel
            for rel in self._outgoing[relation.subject_id]
        ]
        self._incoming[relation.object_id] = [
            updated if rel.subject_id == relation.subject_id and rel.predicate == relation.predicate else rel
            for rel in self._incoming[relation.object_id]
        ]
        return updated

    def entity(self, entity_id: str) -> Entity:
        return self.entities[entity_id]

    def maybe_entity(self, entity_id: str) -> Entity | None:
        return self.entities.get(entity_id)

    def entities_by_type(self, entity_type: EntityType) -> list[Entity]:
        return [entity for entity in self.entities.values() if entity.type == entity_type]

    def relations_from(
        self, subject_id: str, predicate: RelationType | None = None
    ) -> list[Relation]:
        relations = self._outgoing.get(subject_id, [])
        if predicate is None:
            return list(relations)
        return [relation for relation in relations if relation.predicate == predicate]

    def relations_to(self, object_id: str, predicate: RelationType | None = None) -> list[Relation]:
        relations = self._incoming.get(object_id, [])
        if predicate is None:
            return list(relations)
        return [relation for relation in relations if relation.predicate == predicate]

    def find_by_name(self, entity_type: EntityType, name: str) -> Entity | None:
        normalized = normalize_label(name)
        entity_id = self._name_index.get(entity_type, {}).get(normalized)
        if entity_id:
            return self.entities[entity_id]
        matches = self.search(entity_type, name, limit=1)
        return matches[0][0] if matches else None

    def search(
        self, entity_type: EntityType, query: str, limit: int = 5
    ) -> list[tuple[Entity, float]]:
        normalized = normalize_label(query)
        if not normalized:
            return []
        scored: list[tuple[Entity, float]] = []
        for entity in self.entities_by_type(entity_type):
            candidate_names = [entity.name, *entity.aliases]
            score = max(soft_match_score(normalized, candidate) for candidate in candidate_names)
            if score >= 0.6:
                scored.append((entity, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:limit]

    def summary(self) -> dict[str, dict[str, int]]:
        entity_counts: dict[str, int] = defaultdict(int)
        relation_counts: dict[str, int] = defaultdict(int)
        for entity in self.entities.values():
            entity_counts[entity.type.value] += 1
        for relation in self.relations.values():
            relation_counts[relation.predicate.value] += 1
        return {
            "entity_counts": dict(sorted(entity_counts.items())),
            "relation_counts": dict(sorted(relation_counts.items())),
        }

    def _index_entity(self, entity: Entity) -> None:
        for label in [entity.name, *entity.aliases]:
            normalized = normalize_label(label)
            if normalized:
                self._name_index[entity.type][normalized] = entity.id

    def iter_entities(self) -> Iterable[Entity]:
        return self.entities.values()

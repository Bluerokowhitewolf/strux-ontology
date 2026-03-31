from strux.ontology.model import EntityType, RelationType
from strux.runtime import get_runtime


def test_ontology_summary_has_core_structure() -> None:
    summary = get_runtime().graph.summary()
    assert summary["entity_counts"]["career"] >= 1000
    assert summary["entity_counts"]["skill"] >= 400
    assert summary["entity_counts"]["major"] >= 8
    assert summary["relation_counts"]["requiresSkill"] >= 30000
    assert summary["relation_counts"]["hasPrerequisite"] >= 100


def test_software_developers_have_required_skills() -> None:
    graph = get_runtime().graph
    software_developers = graph.find_by_name(EntityType.CAREER, "Software Developers")
    assert software_developers is not None

    required_skill_names = {
        graph.entity(relation.object_id).name
        for relation in graph.relations_from(software_developers.id, RelationType.REQUIRES_SKILL)
    }
    assert "Programming" in required_skill_names


def test_data_scientists_receive_reconstructed_skill_signal() -> None:
    graph = get_runtime().graph
    data_scientists = graph.find_by_name(EntityType.CAREER, "Data Scientists")
    assert data_scientists is not None

    required_skill_names = {
        graph.entity(relation.object_id).name
        for relation in graph.relations_from(data_scientists.id, RelationType.REQUIRES_SKILL)
    }
    assert "Data Analysis and Visualization" in required_skill_names

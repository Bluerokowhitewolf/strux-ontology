from strux.decision.pathing import BridgePathPlanner
from strux.ontology.model import EntityType
from strux.runtime import get_runtime


def test_bridge_path_planner_reaches_machine_learning_from_foundations() -> None:
    graph = get_runtime().graph
    planner = BridgePathPlanner(graph)

    programming = graph.find_by_name(EntityType.SKILL, "Programming")
    mathematics = graph.find_by_name(EntityType.SKILL, "Mathematics")
    machine_learning = graph.find_by_name(EntityType.SKILL, "Machine Learning")

    assert programming is not None
    assert mathematics is not None
    assert machine_learning is not None

    analysis = planner.analyze(
        {
            programming.id: 0.9,
            mathematics.id: 0.85,
        },
        {
            machine_learning.id: 1.0,
        },
    )

    evidence = analysis.skill_paths[machine_learning.id]
    path_names = [graph.entity(node_id).name for node_id in evidence.path_node_ids if node_id in graph.entities]

    assert analysis.bridge_reachability > 0.0
    assert analysis.path_efficiency > 0.0
    assert evidence.bridge_count >= 1
    assert path_names[-1] == "Machine Learning"
    assert any(name in {"Programming", "Mathematics"} for name in path_names[:-1])

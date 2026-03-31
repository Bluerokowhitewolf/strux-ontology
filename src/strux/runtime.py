from dataclasses import dataclass
from functools import lru_cache

from strux.decision.recommender import RecommendationEngine
from strux.interpretation.openai_profile import OpenAIProfileInterpreter
from strux.ontology.builder import OntologyBuilder
from strux.ontology.graph import OntologyGraph


@dataclass(frozen=True)
class StruxRuntime:
    graph: OntologyGraph
    profile_interpreter: OpenAIProfileInterpreter
    recommender: RecommendationEngine


@lru_cache
def get_runtime() -> StruxRuntime:
    graph = OntologyBuilder().build()
    profile_interpreter = OpenAIProfileInterpreter(graph)
    return StruxRuntime(
        graph=graph,
        profile_interpreter=profile_interpreter,
        recommender=RecommendationEngine(graph, profile_interpreter=profile_interpreter),
    )

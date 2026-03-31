from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from strux.decision.models import (
    ActionSimulationReport,
    ActionSimulationRequest,
    RecommendationReport,
    StudentProfile,
    WorkspaceAnalysis,
)
from strux.interpretation.models import ProfileInterpretationResult
from strux.ontology.model import EntityType
from strux.runtime import get_runtime


APP_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))

app = FastAPI(
    title="스트럭스",
    summary="온톨로지 기반 학생 진로 의사결정 지원 시스템",
    description=(
        "학생·진로·역량·전공·활동 사이의 구조를 해석 가능한 형태로 복원하여 "
        "설명 가능한 추천을 제공하는 미니 의미 시스템입니다."
    ),
)
app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")


@app.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"page_title": "스트럭스 진로 의미 구조 탐색기"},
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ontology/summary")
def ontology_summary() -> dict[str, dict[str, int]]:
    return get_runtime().graph.summary()


@app.get("/ontology/careers")
def list_careers(
    query: str | None = Query(default=None, description="직업명 검색어"),
    limit: int = Query(default=20, ge=1, le=100, description="반환 개수"),
) -> list[dict[str, object]]:
    graph = get_runtime().graph
    careers = [
        entity
        for entity in graph.entities_by_type(EntityType.CAREER)
        if entity.attributes.get("profile_kind") == "occupation"
    ]
    if query:
        matches = graph.search(EntityType.CAREER, query, limit=limit)
        careers = [entity for entity, _ in matches if entity.attributes.get("profile_kind") == "occupation"]
    return [
        {
            "id": career.id,
            "name": career.name,
            "job_zone": career.attributes.get("job_zone"),
            "soc_code": career.attributes.get("soc_code"),
        }
        for career in careers[:limit]
    ]


@app.get("/ontology/activities")
def list_activities(
    query: str | None = Query(default=None, description="활동명 검색어"),
    limit: int = Query(default=20, ge=1, le=100, description="반환 개수"),
) -> list[dict[str, object]]:
    graph = get_runtime().graph
    activities = [
        entity
        for entity in graph.entities_by_type(EntityType.ACTIVITY)
        if entity.attributes.get("activity_kind") == "student"
    ]
    if query:
        matches = graph.search(EntityType.ACTIVITY, query, limit=limit)
        activities = [entity for entity, _ in matches if entity.attributes.get("activity_kind") == "student"]
    return [
        {
            "id": activity.id,
            "name": activity.name,
            "description": activity.description,
        }
        for activity in activities[:limit]
    ]


@app.get("/ontology/entities/{entity_id}")
def get_entity(entity_id: str) -> dict[str, object]:
    graph = get_runtime().graph
    entity = graph.maybe_entity(entity_id)
    if entity is None:
        raise HTTPException(status_code=404, detail="엔티티를 찾을 수 없습니다.")
    outgoing = graph.relations_from(entity_id)[:25]
    incoming = graph.relations_to(entity_id)[:25]
    return {
        "entity": entity.model_dump(),
        "outgoing_relations": [relation.model_dump() for relation in outgoing],
        "incoming_relations": [relation.model_dump() for relation in incoming],
    }


@app.post("/interpretations/profile", response_model=ProfileInterpretationResult)
def interpret_profile(profile: StudentProfile) -> ProfileInterpretationResult:
    return get_runtime().profile_interpreter.interpret_profile(profile.model_dump(mode="python"))


@app.post("/recommendations", response_model=RecommendationReport)
def recommend(profile: StudentProfile) -> RecommendationReport:
    return get_runtime().recommender.recommend(profile)


@app.post("/analysis/workspace", response_model=WorkspaceAnalysis)
def analyze_workspace(profile: StudentProfile) -> WorkspaceAnalysis:
    return get_runtime().recommender.analyze_workspace(profile)


@app.post("/actions/simulate-activity", response_model=ActionSimulationReport)
def simulate_activity(payload: ActionSimulationRequest) -> ActionSimulationReport:
    try:
        return get_runtime().recommender.simulate_activity_injection(
            payload.profile,
            payload.activity_name,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

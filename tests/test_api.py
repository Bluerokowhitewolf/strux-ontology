import json
from pathlib import Path

from fastapi.testclient import TestClient

from strux.api.app import app


client = TestClient(app)


def test_root_ui_contains_korean_text() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "학생 진로 의미 구조 탐색기" in response.text


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_profile_interpretation_endpoint_returns_resolved_entities() -> None:
    response = client.post(
        "/interpretations/profile",
        json={
            "skills": ["Programming"],
            "major_interests": ["Computer Science"],
            "activities": ["Research Project"],
        },
    )
    assert response.status_code == 200
    body = response.json()
    names = {item["entity_name"] for item in body["entities"]}
    assert "Programming" in names
    assert "Computer Science" in names
    assert "Research Project" in names


def test_recommendation_endpoint_returns_explanatory_output() -> None:
    payload = json.loads((Path(__file__).resolve().parents[1] / "examples" / "student_profile.json").read_text())
    response = client.post("/recommendations", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["top_careers"][0]["career_name"] == "Data Scientists"
    assert body["top_careers"][0]["fit_label"]
    assert body["top_careers"][0]["score_breakdown"]
    assert body["top_careers"][0]["ontology_paths"]
    assert body["top_careers"][0]["readiness_plan"]["milestone_path"]
    assert body["top_careers"][0]["market_signal"]["market_contexts"]
    assert body["top_careers"][0]["market_signal"]["technology_highlights"]
    assert body["top_majors"][0]["major_name"] == "Data Science"
    assert body["interpreted_state"]["profile_interpretations"]
    assert body["student_readiness"]["current_job_zone"] >= 1
    assert body["student_readiness"]["zone_scores"]


def test_workspace_analysis_endpoint_returns_graph_structure() -> None:
    payload = json.loads((Path(__file__).resolve().parents[1] / "examples" / "student_profile.json").read_text())
    response = client.post("/analysis/workspace", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["report"]["top_careers"][0]["career_name"] == "Data Scientists"
    assert body["workspace"]["nodes"]
    assert body["workspace"]["edges"]
    assert any(node["entity_type"] == "student" for node in body["workspace"]["nodes"])
    assert any(edge["relation"] == "requiresSkill" for edge in body["workspace"]["edges"])


def test_activity_action_endpoint_returns_future_feedback() -> None:
    response = client.post(
        "/actions/simulate-activity",
        json={
            "profile": {
                "name": "Action Student",
                "skills": ["Programming", "Mathematics", "Data Analysis and Visualization"],
                "career_interests": ["Data Science"],
                "major_interests": ["Data Science"],
                "traits": ["Investigative"],
                "top_k": 5,
            },
            "activity_name": "Research Project",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["injected_activity_name"] == "Research Project"
    assert body["action_summary"]
    assert body["future_feedback"]
    assert any(item["skill_name"] == "Machine Learning" for item in body["skill_impacts"])

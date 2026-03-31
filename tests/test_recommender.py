import json
from pathlib import Path

from strux.decision.models import StudentProfile
from strux.runtime import get_runtime


def test_sample_student_prefers_data_science_path() -> None:
    payload = json.loads((Path(__file__).resolve().parents[1] / "examples" / "student_profile.json").read_text())
    profile = StudentProfile.model_validate(payload)

    report = get_runtime().recommender.recommend(profile)

    assert report.top_careers
    assert report.top_careers[0].career_name == "Data Scientists"
    assert report.top_majors
    assert report.top_majors[0].major_name == "Data Science"
    assert "Data Analysis and Visualization" in report.top_careers[0].matched_skills
    assert report.interpreted_state.profile_interpretations
    assert report.student_readiness.current_job_zone >= 1
    assert report.student_readiness.zone_scores
    assert any(item.entity_name == "Data Science" for item in report.interpreted_state.profile_interpretations)
    assert any(component.label == "브리지 도달성" for component in report.top_careers[0].score_breakdown)
    assert report.top_careers[0].readiness_plan.target_job_zone == report.top_careers[0].job_zone
    assert report.top_careers[0].readiness_plan.milestone_path
    assert "Job Zone" in report.top_careers[0].readiness_plan.explanation
    assert report.top_careers[0].market_signal.score >= 0.0
    assert report.top_careers[0].market_signal.market_contexts
    assert report.top_careers[0].market_signal.technology_highlights


def test_activity_simulation_projects_future_delta() -> None:
    profile = StudentProfile.model_validate(
        {
            "name": "Action Student",
            "skills": ["Programming", "Mathematics", "Data Analysis and Visualization"],
            "career_interests": ["Data Science"],
            "major_interests": ["Data Science"],
            "traits": ["Investigative"],
            "top_k": 5,
        }
    )

    report = get_runtime().recommender.simulate_activity_injection(profile, "Research Project")

    assert report.injected_activity_name == "Research Project"
    assert any(impact.skill_name == "Machine Learning" for impact in report.skill_impacts)
    assert any(
        impact.career_name == "Data Scientists" and impact.score_delta > 0
        for impact in report.career_impacts
    )
    assert report.baseline_report.student_readiness.current_job_zone >= 1
    assert report.simulated_report.top_careers[0].readiness_plan.status

from strux.interpretation.normalization import clamp


COMPONENT_LABELS = {
    "skill_alignment": "역량 정합도",
    "bridge_reachability": "브리지 도달성",
    "path_efficiency": "경로 효율성",
    "frontier_signal": "성장 프런티어 신호",
    "interest_alignment": "관심 정합도",
    "activity_alignment": "활동 경로 정합도",
    "major_alignment": "전공 경로 정합도",
    "trait_alignment": "성향 정합도",
    "market_alignment": "시장 신호 정합도",
    "zone_fit": "준비도 적합성",
    "readiness_path": "준비도 경로 정합도",
}


def fit_label(score: float) -> str:
    if score >= 0.45:
        return "강한 적합"
    if score >= 0.30:
        return "성장형 적합"
    return "탐색형 적합"


def component_breakdown(components: dict[str, float]) -> list[dict[str, float | str]]:
    return [
        {"label": COMPONENT_LABELS[key], "score": round(clamp(value), 4)}
        for key, value in components.items()
        if key in COMPONENT_LABELS
    ]


def job_zone_label(job_zone: int | None) -> str:
    if job_zone is None:
        return "준비도 정보 없음"
    if job_zone <= 2:
        return "초기 진입형"
    if job_zone == 3:
        return "중간 준비형"
    return "고준비형"

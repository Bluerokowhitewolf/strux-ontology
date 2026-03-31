import argparse
import json
from pathlib import Path

from strux.decision.models import StudentProfile
from strux.runtime import get_runtime


def main() -> None:
    parser = argparse.ArgumentParser(description="Strux ontology-oriented decision support CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("summary", help="Show ontology summary counts")

    interpret_parser = subparsers.add_parser(
        "interpret",
        help="Interpret a student profile JSON or a free-text input into ontology entities",
    )
    interpret_parser.add_argument(
        "value",
        type=str,
        help="Profile JSON path or free-text input",
    )

    recommend_parser = subparsers.add_parser("recommend", help="Run a recommendation from JSON")
    recommend_parser.add_argument("profile", type=Path, help="Path to a student profile JSON file")

    args = parser.parse_args()
    runtime = get_runtime()

    if args.command == "summary":
        print(json.dumps(runtime.graph.summary(), indent=2, ensure_ascii=False))
        return

    if args.command == "recommend":
        payload = json.loads(args.profile.read_text(encoding="utf-8"))
        profile = StudentProfile.model_validate(payload)
        report = runtime.recommender.recommend(profile)
        print(report.model_dump_json(indent=2))
        return

    if args.command == "interpret":
        candidate_path = Path(args.value)
        if candidate_path.exists():
            payload = json.loads(candidate_path.read_text(encoding="utf-8"))
        else:
            payload = {"free_text": args.value}
        report = runtime.profile_interpreter.interpret_profile(payload)
        print(report.model_dump_json(indent=2, exclude_none=True))

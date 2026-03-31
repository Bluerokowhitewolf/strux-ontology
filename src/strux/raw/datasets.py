from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from strux.settings import get_settings


@dataclass(frozen=True)
class RawDatasetBundle:
    occupations: pd.DataFrame
    skills: pd.DataFrame
    knowledge: pd.DataFrame
    work_activities: pd.DataFrame
    interests: pd.DataFrame
    work_styles: pd.DataFrame
    technology_skills: pd.DataFrame
    job_zones: pd.DataFrame
    job_zone_reference: pd.DataFrame
    related_occupations: pd.DataFrame
    career_dataset: pd.DataFrame
    job_posts: pd.DataFrame


class DatasetRepository:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or get_settings().resolved_data_root
        self._bundle: RawDatasetBundle | None = None

    def load(self) -> RawDatasetBundle:
        if self._bundle is None:
            self._bundle = RawDatasetBundle(
                occupations=self._read_excel("Occupation Data.xlsx"),
                skills=self._read_excel("Skills.xlsx"),
                knowledge=self._read_excel("Knowledge.xlsx"),
                work_activities=self._read_excel("Work Activities.xlsx"),
                interests=self._read_excel("Interests.xlsx"),
                work_styles=self._read_excel("Work Styles.xlsx"),
                technology_skills=self._read_excel("Technology Skills.xlsx"),
                job_zones=self._read_excel("Job Zones.xlsx"),
                job_zone_reference=self._read_excel("Job Zone Reference.xlsx"),
                related_occupations=self._read_excel("Related Occupations.xlsx"),
                career_dataset=self._read_excel("Career Dataset.xlsx"),
                job_posts=self._read_csv("all_job_post.csv"),
            )
        return self._bundle

    def _read_excel(self, filename: str) -> pd.DataFrame:
        return pd.read_excel(self.root / filename)

    def _read_csv(self, filename: str) -> pd.DataFrame:
        return pd.read_csv(self.root / filename)

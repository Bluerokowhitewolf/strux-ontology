from collections import Counter, defaultdict

import pandas as pd

from strux.config.profiles import (
    load_activity_profiles,
    load_major_profiles,
    load_skill_bridges,
)
from strux.interpretation.normalization import (
    clamp,
    coerce_float,
    normalize_label,
    normalize_score,
    slugify,
    soft_match_score,
    split_list_like,
    weighted_overlap,
)
from strux.ontology.graph import OntologyGraph
from strux.ontology.model import Evidence, EvidenceLayer, Entity, EntityType, Relation, RelationType
from strux.ontology.semantics import entity_skill_vector
from strux.raw.datasets import DatasetRepository
from strux.settings import Settings, get_settings


class OntologyBuilder:
    def __init__(
        self,
        repository: DatasetRepository | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.repository = repository or DatasetRepository(self.settings.resolved_data_root)
        self.graph = OntologyGraph()
        self._career_by_code: dict[str, str] = {}

    def build(self) -> OntologyGraph:
        bundle = self.repository.load()

        self._add_occupations(bundle.occupations)
        self._add_onet_skills(bundle.skills)
        self._add_skill_bridges()
        self._add_onet_knowledge(bundle.knowledge)
        self._add_onet_traits(bundle.interests, bundle.work_styles)
        self._add_onet_work_activities(bundle.work_activities)
        self._add_technology_examples(bundle.technology_skills)
        self._add_job_zones(bundle.job_zones, bundle.job_zone_reference)
        self._add_related_occupations(bundle.related_occupations)
        self._add_career_clusters(bundle.career_dataset)
        self._add_market_categories(bundle.job_posts)
        self._link_broad_careers_to_occupations()
        self._propagate_broad_signals_to_occupations()
        self._add_majors()
        self._add_student_activities()
        self._link_majors_to_careers()
        self._link_student_activities_to_careers()
        return self.graph

    def _add_occupations(self, frame: pd.DataFrame) -> None:
        renamed = frame.rename(
            columns={
                "O*NET-SOC Code": "soc_code",
                "Title": "title",
                "Description": "description",
            }
        )
        for row in renamed.itertuples(index=False):
            entity_id = f"career:occupation:{row.soc_code}"
            entity = Entity(
                id=entity_id,
                type=EntityType.CAREER,
                name=str(row.title).strip(),
                description=str(row.description).strip(),
                attributes={
                    "profile_kind": "occupation",
                    "soc_code": str(row.soc_code).strip(),
                },
                provenance=[
                    self._evidence(
                        source="Occupation Data.xlsx",
                        layer=EvidenceLayer.RAW_DATA,
                        locator=f"O*NET-SOC Code={row.soc_code}",
                    )
                ],
            )
            self.graph.add_entity(entity)
            self._career_by_code[str(row.soc_code).strip()] = entity_id

    def _add_onet_skills(self, frame: pd.DataFrame) -> None:
        renamed = frame.rename(
            columns={
                "O*NET-SOC Code": "soc_code",
                "Title": "title",
                "Element Name": "skill_name",
                "Scale ID": "scale_id",
                "Data Value": "data_value",
            }
        )
        renamed = renamed[renamed["scale_id"].isin(["IM", "LV"])].copy()
        renamed["data_value"] = renamed["data_value"].apply(coerce_float)
        pivot = (
            renamed.pivot_table(
                index=["soc_code", "title", "skill_name"],
                columns="scale_id",
                values="data_value",
                aggfunc="mean",
            )
            .reset_index()
            .fillna(0.0)
        )

        for row in pivot.itertuples(index=False):
            career_id = self._career_by_code.get(str(row.soc_code).strip())
            if not career_id:
                continue
            skill_entity = self._ensure_entity(
                entity_type=EntityType.SKILL,
                name=str(row.skill_name).strip(),
                entity_id=f"skill:{slugify(row.skill_name)}",
                source="Skills.xlsx",
                layer=EvidenceLayer.RAW_DATA,
                attributes={"skill_family": "onet"},
            )
            weight = clamp((normalize_score(row.IM, 5.0) * 0.65) + (normalize_score(row.LV, 7.0) * 0.35))
            self.graph.add_relation(
                Relation(
                    subject_id=career_id,
                    predicate=RelationType.REQUIRES_SKILL,
                    object_id=skill_entity.id,
                    weight=weight,
                    attributes={
                        "importance": coerce_float(row.IM),
                        "level": coerce_float(row.LV),
                        "source_kind": "onet-skill",
                    },
                    evidence=[
                        self._evidence(
                            source="Skills.xlsx",
                            layer=EvidenceLayer.RAW_DATA,
                            locator=f"O*NET-SOC Code={row.soc_code}; skill={row.skill_name}",
                            weight=weight,
                        )
                    ],
                )
            )

    def _add_skill_bridges(self) -> None:
        for spec in load_skill_bridges():
            skill = self._ensure_entity(
                entity_type=EntityType.SKILL,
                name=spec["name"],
                entity_id=f"skill:{slugify(spec['name'])}",
                description=spec.get("description"),
                aliases=spec.get("aliases", []),
                source="skill_bridges.yaml",
                layer=EvidenceLayer.INTERPRETATION,
                attributes={"skill_family": "interpretive-bridge"},
            )
            for prerequisite_name, strength in spec.get("prerequisites", {}).items():
                prerequisite = self._ensure_entity(
                    entity_type=EntityType.SKILL,
                    name=prerequisite_name,
                    entity_id=f"skill:{slugify(prerequisite_name)}",
                    source="skill_bridges.yaml",
                    layer=EvidenceLayer.INTERPRETATION,
                    attributes={"skill_family": "foundation"},
                )
                self.graph.add_relation(
                    Relation(
                        subject_id=skill.id,
                        predicate=RelationType.HAS_PREREQUISITE,
                        object_id=prerequisite.id,
                        weight=clamp(coerce_float(strength)),
                        evidence=[
                            self._evidence(
                                source="skill_bridges.yaml",
                                layer=EvidenceLayer.INTERPRETATION,
                                locator=f"skill={spec['name']}; prerequisite={prerequisite_name}",
                            )
                        ],
                    )
                )

    def _add_onet_knowledge(self, frame: pd.DataFrame) -> None:
        renamed = frame.rename(
            columns={
                "O*NET-SOC Code": "soc_code",
                "Element Name": "knowledge_name",
                "Scale ID": "scale_id",
                "Data Value": "data_value",
            }
        )
        renamed = renamed[renamed["scale_id"].isin(["IM", "LV"])].copy()
        renamed["data_value"] = renamed["data_value"].apply(coerce_float)
        pivot = (
            renamed.pivot_table(
                index=["soc_code", "knowledge_name"],
                columns="scale_id",
                values="data_value",
                aggfunc="mean",
            )
            .reset_index()
            .fillna(0.0)
        )

        for row in pivot.itertuples(index=False):
            career_id = self._career_by_code.get(str(row.soc_code).strip())
            if not career_id:
                continue
            knowledge = self._ensure_entity(
                entity_type=EntityType.KNOWLEDGE,
                name=str(row.knowledge_name).strip(),
                entity_id=f"knowledge:{slugify(row.knowledge_name)}",
                source="Knowledge.xlsx",
                layer=EvidenceLayer.RAW_DATA,
            )
            weight = clamp((normalize_score(row.IM, 5.0) * 0.6) + (normalize_score(row.LV, 7.0) * 0.4))
            self.graph.add_relation(
                Relation(
                    subject_id=career_id,
                    predicate=RelationType.REQUIRES_KNOWLEDGE,
                    object_id=knowledge.id,
                    weight=weight,
                    attributes={
                        "importance": coerce_float(row.IM),
                        "level": coerce_float(row.LV),
                    },
                    evidence=[
                        self._evidence(
                            source="Knowledge.xlsx",
                            layer=EvidenceLayer.RAW_DATA,
                            locator=f"O*NET-SOC Code={row.soc_code}; knowledge={row.knowledge_name}",
                            weight=weight,
                        )
                    ],
                )
            )

    def _add_onet_traits(self, interests: pd.DataFrame, work_styles: pd.DataFrame) -> None:
        interests = interests.rename(
            columns={
                "O*NET-SOC Code": "soc_code",
                "Element Name": "trait_name",
                "Data Value": "data_value",
                "Scale ID": "scale_id",
            }
        )
        interests["data_value"] = interests["data_value"].apply(coerce_float)
        for row in interests.itertuples(index=False):
            if str(row.scale_id).strip() != "OI":
                continue
            self._link_trait(
                career_id=self._career_by_code.get(str(row.soc_code).strip()),
                trait_name=str(row.trait_name).strip(),
                weight=normalize_score(row.data_value, 5.0),
                source="Interests.xlsx",
                trait_family="interest",
                locator=f"O*NET-SOC Code={row.soc_code}; trait={row.trait_name}",
            )

        work_styles = work_styles.rename(
            columns={
                "O*NET-SOC Code": "soc_code",
                "Element Name": "trait_name",
                "Data Value": "data_value",
                "Scale ID": "scale_id",
            }
        )
        work_styles["data_value"] = work_styles["data_value"].apply(coerce_float)
        for row in work_styles.itertuples(index=False):
            if str(row.scale_id).strip() != "WI":
                continue
            self._link_trait(
                career_id=self._career_by_code.get(str(row.soc_code).strip()),
                trait_name=str(row.trait_name).strip(),
                weight=normalize_score(row.data_value, 5.0),
                source="Work Styles.xlsx",
                trait_family="work-style",
                locator=f"O*NET-SOC Code={row.soc_code}; trait={row.trait_name}",
            )

    def _add_onet_work_activities(self, frame: pd.DataFrame) -> None:
        renamed = frame.rename(
            columns={
                "O*NET-SOC Code": "soc_code",
                "Element Name": "activity_name",
                "Scale ID": "scale_id",
                "Data Value": "data_value",
            }
        )
        renamed = renamed[renamed["scale_id"].isin(["IM", "LV"])].copy()
        renamed["data_value"] = renamed["data_value"].apply(coerce_float)
        pivot = (
            renamed.pivot_table(
                index=["soc_code", "activity_name"],
                columns="scale_id",
                values="data_value",
                aggfunc="mean",
            )
            .reset_index()
            .fillna(0.0)
        )

        for row in pivot.itertuples(index=False):
            career_id = self._career_by_code.get(str(row.soc_code).strip())
            if not career_id:
                continue
            activity = self._ensure_entity(
                entity_type=EntityType.ACTIVITY,
                name=str(row.activity_name).strip(),
                entity_id=f"activity:{slugify(row.activity_name)}",
                source="Work Activities.xlsx",
                layer=EvidenceLayer.RAW_DATA,
                attributes={"activity_kind": "occupational"},
            )
            weight = clamp((normalize_score(row.IM, 5.0) * 0.6) + (normalize_score(row.LV, 7.0) * 0.4))
            self.graph.add_relation(
                Relation(
                    subject_id=career_id,
                    predicate=RelationType.INVOLVES_ACTIVITY,
                    object_id=activity.id,
                    weight=weight,
                    evidence=[
                        self._evidence(
                            source="Work Activities.xlsx",
                            layer=EvidenceLayer.RAW_DATA,
                            locator=f"O*NET-SOC Code={row.soc_code}; activity={row.activity_name}",
                        )
                    ],
                )
            )

    def _add_job_zones(self, job_zones: pd.DataFrame, references: pd.DataFrame) -> None:
        references = references.rename(
            columns={
                "Job Zone": "job_zone",
                "Name": "zone_name",
                "Experience": "experience",
                "Education": "education",
                "Job Training": "job_training",
                "SVP Range": "svp_range",
            }
        ).copy()
        reference_map = {
            str(row.job_zone).strip(): {
                "name": str(row.zone_name).strip(),
                "experience": str(row.experience).strip(),
                "education": str(row.education).strip(),
                "job_training": str(row.job_training).strip(),
                "svp_range": str(row.svp_range).strip(),
            }
            for row in references.itertuples(index=False)
        }

        zones = job_zones.rename(columns={"O*NET-SOC Code": "soc_code", "Job Zone": "job_zone"})
        for row in zones.itertuples(index=False):
            career_id = self._career_by_code.get(str(row.soc_code).strip())
            if not career_id:
                continue
            career = self.graph.entity(career_id)
            zone_value = str(row.job_zone).strip()
            reference = reference_map.get(zone_value, {})
            updated = career.model_copy(
                update={
                    "attributes": {
                        **career.attributes,
                        "job_zone": int(zone_value) if zone_value.isdigit() else None,
                        "job_zone_reference": reference,
                    },
                    "provenance": [
                        *career.provenance,
                        self._evidence(
                            source="Job Zones.xlsx",
                            layer=EvidenceLayer.RAW_DATA,
                            locator=f"O*NET-SOC Code={row.soc_code}",
                        ),
                    ],
                }
            )
            self.graph.add_entity(updated)

    def _add_technology_examples(self, frame: pd.DataFrame) -> None:
        renamed = frame.rename(
            columns={
                "O*NET-SOC Code": "soc_code",
                "Example": "example",
                "Commodity Title": "commodity_title",
                "Hot Technology": "hot_technology",
                "In Demand": "in_demand",
            }
        )
        grouped = defaultdict(list)
        for row in renamed.itertuples(index=False):
            soc_code = str(row.soc_code).strip()
            if soc_code not in self._career_by_code:
                continue
            grouped[soc_code].append(
                {
                    "example": str(row.example).strip(),
                    "commodity_title": str(row.commodity_title).strip(),
                    "hot_technology": str(row.hot_technology).strip() == "Y",
                    "in_demand": str(row.in_demand).strip() == "Y",
                }
            )

        for soc_code, items in grouped.items():
            career_id = self._career_by_code[soc_code]
            career = self.graph.entity(career_id)
            prioritized = sorted(items, key=lambda item: (item["in_demand"], item["hot_technology"]), reverse=True)
            updated = career.model_copy(
                update={
                    "attributes": {
                        **career.attributes,
                        "technology_examples": prioritized[:12],
                    },
                    "provenance": [
                        *career.provenance,
                        self._evidence(
                            source="Technology Skills.xlsx",
                            layer=EvidenceLayer.RAW_DATA,
                            locator=f"O*NET-SOC Code={soc_code}",
                        ),
                    ],
                }
            )
            self.graph.add_entity(updated)

    def _add_related_occupations(self, frame: pd.DataFrame) -> None:
        renamed = frame.rename(
            columns={
                "O*NET-SOC Code": "soc_code",
                "Related O*NET-SOC Code": "related_soc_code",
                "Relatedness Tier": "tier",
            }
        )
        for row in renamed.itertuples(index=False):
            source_id = self._career_by_code.get(str(row.soc_code).strip())
            target_id = self._career_by_code.get(str(row.related_soc_code).strip())
            if not source_id or not target_id:
                continue
            tier = str(row.tier).strip()
            weight = 1.0 if "Primary" in tier else 0.7
            self.graph.add_relation(
                Relation(
                    subject_id=source_id,
                    predicate=RelationType.RELATED_TO_CAREER,
                    object_id=target_id,
                    weight=weight,
                    attributes={"tier": tier},
                    evidence=[
                        self._evidence(
                            source="Related Occupations.xlsx",
                            layer=EvidenceLayer.RAW_DATA,
                            locator=f"O*NET-SOC Code={row.soc_code}; related={row.related_soc_code}",
                        )
                    ],
                )
            )

    def _add_career_clusters(self, frame: pd.DataFrame) -> None:
        skill_counts_by_career: dict[str, Counter[str]] = defaultdict(Counter)
        for row in frame.itertuples(index=False):
            career_name = str(getattr(row, "Career")).strip()
            for skill_name in split_list_like(getattr(row, "Skill")):
                skill_counts_by_career[career_name][skill_name] += 1

        for career_name, counter in skill_counts_by_career.items():
            entity = self._ensure_entity(
                entity_type=EntityType.CAREER,
                name=career_name,
                entity_id=f"career:cluster:{slugify(career_name)}",
                source="Career Dataset.xlsx",
                layer=EvidenceLayer.RAW_DATA,
                attributes={
                    "profile_kind": "career_cluster",
                    "record_count": sum(counter.values()),
                },
            )
            peak = max(counter.values()) if counter else 1
            for skill_name, count in counter.items():
                skill = self._ensure_skill_from_external(skill_name, "Career Dataset.xlsx")
                weight = clamp(0.2 + (count / peak) * 0.8)
                self.graph.add_relation(
                    Relation(
                        subject_id=entity.id,
                        predicate=RelationType.REQUIRES_SKILL,
                        object_id=skill.id,
                        weight=weight,
                        attributes={"occurrences": count, "source_kind": "career-dataset"},
                        evidence=[
                            self._evidence(
                                source="Career Dataset.xlsx",
                                layer=EvidenceLayer.RAW_DATA,
                                locator=f"Career={career_name}; skill={skill_name}",
                                weight=weight,
                            )
                        ],
                    )
                )

    def _add_market_categories(self, frame: pd.DataFrame) -> None:
        skill_counts: dict[str, Counter[str]] = defaultdict(Counter)
        title_counts: dict[str, Counter[str]] = defaultdict(Counter)

        for row in frame.itertuples(index=False):
            category = str(getattr(row, "category")).strip()
            title = str(getattr(row, "job_title")).strip()
            title_counts[category][title] += 1
            for skill_name in split_list_like(getattr(row, "job_skill_set")):
                skill_counts[category][skill_name] += 1

        for category, counter in skill_counts.items():
            entity = self._ensure_entity(
                entity_type=EntityType.CAREER,
                name=category,
                entity_id=f"career:market:{slugify(category)}",
                source="all_job_post.csv",
                layer=EvidenceLayer.RAW_DATA,
                attributes={
                    "profile_kind": "market_sector",
                    "post_count": sum(title_counts[category].values()),
                    "sample_titles": [title for title, _ in title_counts[category].most_common(5)],
                },
            )
            filtered_items = [(skill, count) for skill, count in counter.most_common(30) if count >= 2]
            peak = filtered_items[0][1] if filtered_items else 1
            for skill_name, count in filtered_items:
                skill = self._ensure_skill_from_external(skill_name, "all_job_post.csv")
                weight = clamp(0.2 + (count / peak) * 0.8)
                self.graph.add_relation(
                    Relation(
                        subject_id=entity.id,
                        predicate=RelationType.REQUIRES_SKILL,
                        object_id=skill.id,
                        weight=weight,
                        attributes={"occurrences": count, "source_kind": "market-signal"},
                        evidence=[
                            self._evidence(
                                source="all_job_post.csv",
                                layer=EvidenceLayer.RAW_DATA,
                                locator=f"category={category}; skill={skill_name}",
                                weight=weight,
                            )
                        ],
                    )
                )

    def _link_broad_careers_to_occupations(self) -> None:
        occupations = [
            entity
            for entity in self.graph.entities_by_type(EntityType.CAREER)
            if entity.attributes.get("profile_kind") == "occupation"
        ]
        broad_careers = [
            entity
            for entity in self.graph.entities_by_type(EntityType.CAREER)
            if entity.attributes.get("profile_kind") in {"career_cluster", "market_sector"}
        ]

        occupation_vectors = {
            entity.id: entity_skill_vector(self.graph, entity.id, (RelationType.REQUIRES_SKILL,))
            for entity in occupations
        }

        for broad in broad_careers:
            source_vector = entity_skill_vector(self.graph, broad.id, (RelationType.REQUIRES_SKILL,))
            if not source_vector:
                continue
            scored: list[tuple[str, float]] = []
            for occupation in occupations:
                title_score = soft_match_score(broad.name, occupation.name)
                broad_tokens = set(normalize_label(broad.name).split())
                occupation_tokens = set(normalize_label(occupation.name).split())
                shared_tokens = broad_tokens & occupation_tokens
                score = weighted_overlap(source_vector, occupation_vectors[occupation.id])
                if occupation_vectors[occupation.id]:
                    score = clamp(score + (0.15 * title_score))
                elif shared_tokens or normalize_label(broad.name) in normalize_label(occupation.name):
                    score = clamp(max(score, title_score * 0.85))
                if score >= self.settings.cluster_link_threshold:
                    scored.append((occupation.id, score))
            scored.sort(key=lambda item: item[1], reverse=True)
            for occupation_id, score in scored[:8]:
                self.graph.add_relation(
                    Relation(
                        subject_id=broad.id,
                        predicate=RelationType.SPECIALIZES_TO,
                        object_id=occupation_id,
                        weight=score,
                        evidence=[
                            self._evidence(
                                source="interpretation",
                                layer=EvidenceLayer.INTERPRETATION,
                                locator=f"{broad.id}->{occupation_id}",
                                note="Linked by expanded skill overlap and title similarity.",
                                weight=score,
                            )
                        ],
                    )
                )

    def _add_majors(self) -> None:
        for spec in load_major_profiles():
            major = self._ensure_entity(
                entity_type=EntityType.MAJOR,
                name=spec["name"],
                entity_id=f"major:{slugify(spec['name'])}",
                description=spec.get("description"),
                source="major_profiles.yaml",
                layer=EvidenceLayer.INTERPRETATION,
            )
            for skill_name, weight in spec.get("develops_skills", {}).items():
                skill = self._ensure_skill_from_external(skill_name, "major_profiles.yaml")
                self.graph.add_relation(
                    Relation(
                        subject_id=major.id,
                        predicate=RelationType.DEVELOPS_SKILL,
                        object_id=skill.id,
                        weight=clamp(coerce_float(weight)),
                        evidence=[
                            self._evidence(
                                source="major_profiles.yaml",
                                layer=EvidenceLayer.INTERPRETATION,
                                locator=f"major={spec['name']}; skill={skill_name}",
                            )
                        ],
                    )
                )
            for target_name in spec.get("career_targets", []):
                target = self._resolve_career(target_name)
                if target is None:
                    continue
                self.graph.add_relation(
                    Relation(
                        subject_id=major.id,
                        predicate=RelationType.LEADS_TO,
                        object_id=target.id,
                        weight=0.9,
                        evidence=[
                            self._evidence(
                                source="major_profiles.yaml",
                                layer=EvidenceLayer.INTERPRETATION,
                                locator=f"major={spec['name']}; career={target_name}",
                            )
                        ],
                    )
                )

    def _propagate_broad_signals_to_occupations(self) -> None:
        occupations = [
            entity
            for entity in self.graph.entities_by_type(EntityType.CAREER)
            if entity.attributes.get("profile_kind") == "occupation"
        ]
        for occupation in occupations:
            direct_skill_relations = self.graph.relations_from(occupation.id, RelationType.REQUIRES_SKILL)
            incoming_links = sorted(
                self.graph.relations_to(occupation.id, RelationType.SPECIALIZES_TO),
                key=lambda relation: relation.weight,
                reverse=True,
            )
            if not incoming_links:
                continue
            propagation_factor = 0.7 if len(direct_skill_relations) == 0 else 0.3
            for link in incoming_links[:3]:
                source = self.graph.entity(link.subject_id)
                for relation in self.graph.relations_from(source.id, RelationType.REQUIRES_SKILL):
                    propagated_weight = clamp(relation.weight * link.weight * propagation_factor)
                    if propagated_weight < 0.15:
                        continue
                    self.graph.add_relation(
                        Relation(
                            subject_id=occupation.id,
                            predicate=RelationType.REQUIRES_SKILL,
                            object_id=relation.object_id,
                            weight=propagated_weight,
                            attributes={
                                "source_kind": "propagated-broad-signal",
                                "propagated_from": source.id,
                            },
                            evidence=[
                                self._evidence(
                                    source="interpretation",
                                    layer=EvidenceLayer.INTERPRETATION,
                                    locator=f"{source.id}->{occupation.id}; skill={relation.object_id}",
                                    note="Broad career signal propagated into occupation due sparse direct skill evidence.",
                                    weight=propagated_weight,
                                )
                            ],
                        )
                    )

    def _add_student_activities(self) -> None:
        for spec in load_activity_profiles():
            activity = self._ensure_entity(
                entity_type=EntityType.ACTIVITY,
                name=spec["name"],
                entity_id=f"activity:{slugify(spec['name'])}",
                description=spec.get("description"),
                source="activity_profiles.yaml",
                layer=EvidenceLayer.INTERPRETATION,
                attributes={"activity_kind": "student"},
            )
            for skill_name, weight in spec.get("develops_skills", {}).items():
                skill = self._ensure_skill_from_external(skill_name, "activity_profiles.yaml")
                self.graph.add_relation(
                    Relation(
                        subject_id=activity.id,
                        predicate=RelationType.DEVELOPS_SKILL,
                        object_id=skill.id,
                        weight=clamp(coerce_float(weight)),
                        evidence=[
                            self._evidence(
                                source="activity_profiles.yaml",
                                layer=EvidenceLayer.INTERPRETATION,
                                locator=f"activity={spec['name']}; skill={skill_name}",
                            )
                        ],
                    )
                )
            for major_name in spec.get("supports_majors", []):
                major = self.graph.find_by_name(EntityType.MAJOR, major_name)
                if major is None:
                    continue
                self.graph.add_relation(
                    Relation(
                        subject_id=activity.id,
                        predicate=RelationType.SUPPORTS_MAJOR,
                        object_id=major.id,
                        weight=0.8,
                        evidence=[
                            self._evidence(
                                source="activity_profiles.yaml",
                                layer=EvidenceLayer.INTERPRETATION,
                                locator=f"activity={spec['name']}; major={major_name}",
                            )
                        ],
                    )
                )
            for career_name in spec.get("supports_careers", []):
                career = self._resolve_career(career_name)
                if career is None:
                    continue
                self.graph.add_relation(
                    Relation(
                        subject_id=activity.id,
                        predicate=RelationType.SUPPORTS_CAREER,
                        object_id=career.id,
                        weight=0.8,
                        evidence=[
                            self._evidence(
                                source="activity_profiles.yaml",
                                layer=EvidenceLayer.INTERPRETATION,
                                locator=f"activity={spec['name']}; career={career_name}",
                            )
                        ],
                    )
                )

    def _link_majors_to_careers(self) -> None:
        majors = self.graph.entities_by_type(EntityType.MAJOR)
        occupations = [
            entity
            for entity in self.graph.entities_by_type(EntityType.CAREER)
            if entity.attributes.get("profile_kind") == "occupation"
        ]
        occupation_vectors = {
            entity.id: entity_skill_vector(self.graph, entity.id, (RelationType.REQUIRES_SKILL,))
            for entity in occupations
        }
        for major in majors:
            major_vector = entity_skill_vector(self.graph, major.id, (RelationType.DEVELOPS_SKILL,))
            scored: list[tuple[str, float]] = []
            for occupation in occupations:
                score = weighted_overlap(major_vector, occupation_vectors[occupation.id])
                if score >= self.settings.major_link_threshold:
                    scored.append((occupation.id, score))
            scored.sort(key=lambda item: item[1], reverse=True)
            for occupation_id, score in scored[:10]:
                self.graph.add_relation(
                    Relation(
                        subject_id=major.id,
                        predicate=RelationType.LEADS_TO,
                        object_id=occupation_id,
                        weight=score,
                        evidence=[
                            self._evidence(
                                source="interpretation",
                                layer=EvidenceLayer.INTERPRETATION,
                                locator=f"{major.id}->{occupation_id}",
                                note="Linked by expanded skill overlap.",
                                weight=score,
                            )
                        ],
                    )
                )

    def _link_student_activities_to_careers(self) -> None:
        activities = [
            entity
            for entity in self.graph.entities_by_type(EntityType.ACTIVITY)
            if entity.attributes.get("activity_kind") == "student"
        ]
        occupations = [
            entity
            for entity in self.graph.entities_by_type(EntityType.CAREER)
            if entity.attributes.get("profile_kind") == "occupation"
        ]
        occupation_vectors = {
            entity.id: entity_skill_vector(self.graph, entity.id, (RelationType.REQUIRES_SKILL,))
            for entity in occupations
        }
        for activity in activities:
            activity_vector = entity_skill_vector(self.graph, activity.id, (RelationType.DEVELOPS_SKILL,))
            scored: list[tuple[str, float]] = []
            for occupation in occupations:
                score = weighted_overlap(activity_vector, occupation_vectors[occupation.id])
                if score >= self.settings.activity_link_threshold:
                    scored.append((occupation.id, score))
            scored.sort(key=lambda item: item[1], reverse=True)
            for occupation_id, score in scored[:8]:
                self.graph.add_relation(
                    Relation(
                        subject_id=activity.id,
                        predicate=RelationType.SUPPORTS_CAREER,
                        object_id=occupation_id,
                        weight=score,
                        evidence=[
                            self._evidence(
                                source="interpretation",
                                layer=EvidenceLayer.INTERPRETATION,
                                locator=f"{activity.id}->{occupation_id}",
                                note="Linked by expanded skill overlap.",
                                weight=score,
                            )
                        ],
                    )
                )

    def _link_trait(
        self,
        *,
        career_id: str | None,
        trait_name: str,
        weight: float,
        source: str,
        trait_family: str,
        locator: str,
    ) -> None:
        if not career_id:
            return
        trait = self._ensure_entity(
            entity_type=EntityType.TRAIT,
            name=trait_name,
            entity_id=f"trait:{slugify(trait_name)}",
            source=source,
            layer=EvidenceLayer.RAW_DATA,
            attributes={"trait_family": trait_family},
        )
        self.graph.add_relation(
            Relation(
                subject_id=career_id,
                predicate=RelationType.ALIGNED_WITH_TRAIT,
                object_id=trait.id,
                weight=weight,
                attributes={"trait_family": trait_family},
                evidence=[
                    self._evidence(
                        source=source,
                        layer=EvidenceLayer.RAW_DATA,
                        locator=locator,
                        weight=weight,
                    )
                ],
            )
        )

    def _resolve_career(self, name: str) -> Entity | None:
        direct = self.graph.find_by_name(EntityType.CAREER, name)
        if direct:
            return direct
        candidates = self.graph.search(EntityType.CAREER, name, limit=1)
        return candidates[0][0] if candidates else None

    def _ensure_skill_from_external(self, name: str, source: str) -> Entity:
        existing = self.graph.find_by_name(EntityType.SKILL, name)
        if existing:
            return existing
        normalized = normalize_label(name)
        candidates = self.graph.search(EntityType.SKILL, name, limit=3)
        if candidates and candidates[0][1] >= 0.86:
            return candidates[0][0]
        return self._ensure_entity(
            entity_type=EntityType.SKILL,
            name=name,
            entity_id=f"skill:{slugify(name)}",
            source=source,
            layer=EvidenceLayer.INTERPRETATION,
            attributes={"skill_family": "supplemental", "normalized_name": normalized},
        )

    def _ensure_entity(
        self,
        *,
        entity_type: EntityType,
        name: str,
        entity_id: str,
        source: str,
        layer: EvidenceLayer,
        description: str | None = None,
        aliases: list[str] | None = None,
        attributes: dict[str, object] | None = None,
    ) -> Entity:
        entity = Entity(
            id=entity_id,
            type=entity_type,
            name=name,
            description=description,
            aliases=aliases or [],
            attributes=attributes or {},
            provenance=[self._evidence(source=source, layer=layer, locator=f"name={name}")],
        )
        return self.graph.add_entity(entity)

    def _evidence(
        self,
        *,
        source: str,
        layer: EvidenceLayer,
        locator: str | None = None,
        note: str | None = None,
        weight: float | None = None,
    ) -> Evidence:
        return Evidence(source=source, layer=layer, locator=locator, note=note, weight=weight)

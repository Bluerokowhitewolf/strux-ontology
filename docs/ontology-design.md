# Strux Ontology Design

## 1. Problem Framing

The system is designed around a strict distinction:

- reality contains students, careers, skills, majors, activities, traits, and knowledge
- datasets only contain partial records about them

Therefore, the system must not confuse the data layer with the semantic layer.

## 2. Layered Model

### Raw Data Layer

This layer contains only observed records:

- spreadsheet rows
- CSV rows
- scores
- labels
- text fragments

At this stage there is no claim that a field already equals a stable semantic object.

### Interpretation Layer

This layer performs semantic repair and alignment:

- normalizes labels
- resolves aliases and noisy phrases
- bridges domain skills to foundational O*NET skills
- creates explicit seed profiles where raw data is absent

In this project, `Major` and student-side `Activity` are not directly observed in the supplied datasets. They are introduced here as interpretation artifacts.

### Ontology Layer

This layer contains entities and typed relations.

Core entities:

- `Student`
- `Career`
- `Skill`
- `Major`
- `Activity`
- `Trait`
- `Knowledge`

Core relations:

- `Student hasSkill Skill`
- `Student hasInterestIn Career`
- `Student hasInterestIn Major`
- `Student hasTrait Trait`
- `Career requiresSkill Skill`
- `Career requiresKnowledge Knowledge`
- `Career alignedWithTrait Trait`
- `Career involvesActivity Activity`
- `Career relatedToCareer Career`
- `Skill hasPrerequisite Skill`
- `Major developsSkill Skill`
- `Major leadsTo Career`
- `Activity developsSkill Skill`
- `Activity supportsMajor Major`
- `Activity supportsCareer Career`

### Output / Decision Support Layer

This layer does not emit opaque recommendations.

It emits:

- career candidates
- matched skills
- missing skills
- supporting majors
- suggested activities
- ontology path explanations

## 3. Why Skill Is the Central Pivot

The supplied raw datasets do not represent every entity with equal directness.

- `Career` is strongly represented in O*NET.
- `Skill` is strongly represented in O*NET and the career dataset.
- `Knowledge` and `Trait` are moderately represented.
- `Major` is weakly represented.
- student-side `Activity` is weakly represented.

That makes `Skill` the most stable cross-source pivot for the first prototype.

## 4. Why Bridge Files Exist

The raw data contains many domain-facing skill phrases such as:

- web development
- machine learning
- cloud computing
- cybersecurity

O*NET often represents more foundational competencies such as:

- programming
- critical thinking
- systems analysis
- complex problem solving

These vocabularies are not identical. A semantic bridge is required if the system is to reason across them without pretending they are the same thing.

The bridge files make this mediation explicit.

## 5. Treatment of Majors

The project requirement includes `Major` as a first-class ontology entity, but the supplied datasets do not directly define a major ontology.

So the system introduces majors as operational semantic profiles:

- each major develops a weighted bundle of skills
- each major can lead to broad career tracks and concrete occupations
- these links are partly explicit seeds and partly inferred from skill overlap

This is faithful to the stated philosophy: majors are not raw fields here, but interpreted entities.

## 6. Recommendation Logic

Career ranking is based on several explicit components:

- student skill alignment with career skill requirements
- student career-interest alignment
- student trait alignment
- job-zone fit
- market support from the job-post dataset

Major and activity recommendations are generated from the gap between the student state and the target career structure.

This keeps the system explanatory instead of merely predictive.

## 7. Epistemic Limits

This prototype is intentionally honest about what it does not know.

- It does not claim that a recommended career is destiny.
- It does not claim that seed majors are exhaustive.
- It does not claim that raw labels are clean.
- It does not delegate the ontology to an LLM.

Its purpose is to demonstrate a disciplined, operational semantics for student career guidance.

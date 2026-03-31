# Strux

Strux is a mini ontology-based decision support system for student career guidance.

Its goal is not to treat rows, files, or scores as the essence of the problem. It treats them as raw signals that must be interpreted into entities and relations about the real world:

- `Student`
- `Career`
- `Skill`
- `Major`
- `Activity`
- `Trait`
- `Knowledge`

The system is structured in four layers:

1. `Raw Data Layer`
2. `Interpretation Layer`
3. `Ontology Layer`
4. `Output / Decision Support Layer`

## What This Prototype Does

- Loads O*NET occupation, skill, knowledge, interest, work style, work activity, related occupation, and job zone data.
- Loads the provided career-skill dataset and job-post dataset as additional empirical signals.
- Builds an ontology graph with explicit entities and typed relations.
- Separates raw observations from interpreted semantic structures.
- Accepts a student profile and returns:
  - interpretable career recommendations
  - major recommendations
  - activity recommendations
  - matched skills, missing skills, and ontology paths

## Architectural Position

Internally, Strux follows an ontology-first view:

- data is fragmentary
- entities and relations are the operational core
- recommendations are only valid when they can be explained through structure

Externally, it behaves like a recommendation system. Internally, it is a semantic interpretation system.

## Data Coverage

Directly grounded in the supplied files:

- `Occupation Data.xlsx`
- `Skills.xlsx`
- `Knowledge.xlsx`
- `Work Activities.xlsx`
- `Interests.xlsx`
- `Work Styles.xlsx`
- `Technology Skills.xlsx`
- `Job Zones.xlsx`
- `Job Zone Reference.xlsx`
- `Related Occupations.xlsx`
- `Career Dataset.xlsx`
- `all_job_post.csv`

Interpretive seed files added by this prototype:

- `config/skill_bridges.yaml`
- `config/major_profiles.yaml`
- `config/activity_profiles.yaml`

These do not pretend to be raw truth. They are explicit interpretation artifacts used to bridge missing semantics that the raw datasets do not directly provide, especially for `Major` and student-side `Activity`.

## Run

```bash
cd /Users/seungkyuroh/Desktop/Strux
source .venv/bin/activate
python -m pip install -e ".[dev]"
export OPENAI_API_KEY="your-api-key"
PYTHONPATH=src uvicorn strux.api.app:app --reload
```

Open:

- `http://127.0.0.1:8000/`
- `http://127.0.0.1:8000/docs`

## CLI

Ontology summary:

```bash
cd /Users/seungkyuroh/Desktop/Strux
source .venv/bin/activate
PYTHONPATH=src python -m strux summary
```

Demo recommendation:

```bash
cd /Users/seungkyuroh/Desktop/Strux
source .venv/bin/activate
PYTHONPATH=src python -m strux recommend examples/student_profile.json
```

Profile interpretation:

```bash
cd /Users/seungkyuroh/Desktop/Strux
source .venv/bin/activate
export OPENAI_API_KEY="your-api-key"
PYTHONPATH=src python -m strux interpret examples/student_profile.json
```

## API Endpoints

- `GET /health`
- `GET /ontology/summary`
- `GET /ontology/activities`
- `GET /ontology/careers`
- `GET /ontology/entities/{entity_id}`
- `POST /interpretations/profile`
- `POST /recommendations`
- `POST /actions/simulate-activity`

## Philosophy in Code

The code intentionally keeps four concerns separate:

- raw loaders only read files
- interpretation code normalizes and bridges ambiguous data
- ontology code defines entities and relations
- recommendation code never reads spreadsheets directly

This separation is the main design commitment of the project.

## OpenAI Interpretation Layer

The system now uses OpenAI as the interpretation layer for free-form and mixed-language student inputs.

- Strategy: local exact match first, then OpenAI semantic normalization, then ontology entity resolution
- Input fields interpreted together: `free_text`, `skills`, `activities`, `career_interests`, `major_interests`, `traits`
- Output form: English ontology-ready labels such as `Computer Science`, `Programming`, `Research Project`
- Cost control: only unresolved inputs go to the model, output is schema-constrained, and prompt caching keys are configured

Example:

- input: `수학적 사고로 알고리즘 문제를 해결하는 것이 즐겁다`
- normalized entities:
  - `Mathematics`
  - `Programming`
  - `Critical Thinking`

This keeps the ontology layer grounded in canonical English entities while letting students write naturally.

## Action Simulation Layer

The system now includes an action-style simulation layer for future-state exploration.

- Input: current `StudentProfile` plus one candidate student `Activity`
- Mechanism: inject the activity as a virtual activation into the student-side ontology state, propagate its effects through `developsSkill` and prerequisite bridges, then recompute recommendations
- Output: baseline vs simulated career ranking, skill deltas, and future-oriented feedback

Example:

- `Research Project` injected into a data-oriented student state
- projected effect:
  - `Machine Learning` skill improves
  - `Data Scientists` fit score increases

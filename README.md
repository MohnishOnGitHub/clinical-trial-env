---
title: clinical_trial_env
emoji: "🧪"
colorFrom: blue
colorTo: purple
sdk: docker
---

# ClinicalTrialEnv

A multi-turn reinforcement learning environment where an AI agent interviews
a patient by asking questions one at a time, then decides whether they qualify
for a clinical drug trial.

The agent does NOT see the full patient profile upfront. It must ask for fields
one by one (age, eGFR, HbA1c, medications, conditions), then submit a final
eligible/not eligible decision. The agent is penalised for every question asked,
incentivising efficient information gathering.

Built for the Meta PyTorch OpenEnv Hackathon 2026.

---

## Action Space

Two action types:

**ask** request one patient field

```json
{
  "action_type": "ask",
  "field_request": "egfr"
}
```

Valid field names: `age`, `egfr`, `hba1c`, `medications`, `conditions`

**decide** submit final eligibility ruling

```json
{
  "action_type": "decide",
  "eligible": false,
  "reason": "eGFR 38 is below the minimum threshold of 45"
}
```

---

## Observation Space

| Field             | Type | Description                       |
| ----------------- | ---- | --------------------------------- |
| `revealed_fields` | dict | Patient fields revealed so far    |
| `trial_criteria`  | dict | Eligibility rules always visible  |
| `questions_asked` | int  | Running count of questions used   |
| `last_answer`     | str  | Answer to the last question asked |
| `task_id`         | str  | Which task is running             |
| `decision_made`   | bool | Whether episode has ended         |

---

## Tasks

### single_criterion (easy)

One rule: patient age must be between 18 and 65.  
Optimal strategy: ask age, decide immediately. One question maximum.

### multi_criteria (medium)

Four rules: age 3070, eGFR 45, HbA1c 8.0, no warfarin or insulin.  
Partial credit per criterion. Optimal strategy: ask most likely disqualifier first.

### edge_case (hard)

Multiple criteria with borderline values. eGFR may be exactly 44, 45, or 46.  
HbA1c range is 6.59.5. Agent must reason carefully about threshold edge cases.

---

## Reward Structure

| Event                                   | Reward |
| --------------------------------------- | ------ |
| Each question asked                     | -1     |
| Correct final decision                  | +20    |
| Wrong final decision                    | -20    |
| Reason cites specific criterion value   | +2     |
| Hitting max steps (10) without deciding | -5     |

Final score normalised to [0.0, 1.0].

---

## Setup

```bash
pip install openenv-core
cd clinical_trial_env
uv run inference.py
```

Environment variables required:

```bash
HF_TOKEN=hf_your_token_here          # required, no default
API_BASE_URL=https://router.huggingface.co/v1   # has default
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct            # has default
```

---

## Baseline Scores

Scores produced by the rule-based baseline agent in `inference.py`:

| Task             | Steps | Score |
| ---------------- | ----- | ----- |
| single_criterion | 2     | 0.95  |
| multi_criteria   | 4     | 0.98  |
| edge_case        | 2     | 0.98  |

---

## Project Structure

```
clinical_trial_env/
 inference.py          # baseline agent script (must be in root)
 models.py             # ScreeningAction, ScreeningObservation
 tasks.py              # 3 task definitions + graders
 openenv.yaml          # metadata + task IDs
 pyproject.toml
 server/
     clinical_trial_env_environment.py
     app.py
     Dockerfile
```

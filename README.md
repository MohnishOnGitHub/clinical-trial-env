---
title: clinical_trial_env
emoji: 🧪
colorFrom: blue
colorTo: green
sdk: docker
---

# ClinicalTrialEnv

> A multi-turn reinforcement learning environment for clinical trial patient screening — built for the Meta PyTorch OpenEnv Hackathon 2026.

**[Live Demo](https://mohonhf-clinical-trial-env.hf.space/web)** · **[HF Space](https://huggingface.co/spaces/MOHONHF/clinical-trial-env)** · **[GitHub](https://github.com/MohnishOnGitHub/clinical-trial-env)**

---

## What Is This?

ClinicalTrialEnv is an RL environment where an AI agent must determine whether a patient qualifies for a clinical drug trial — but it cannot see the patient's full profile upfront.

The agent must **interview the patient one field at a time**, asking for age, eGFR, HbA1c, medications, or conditions. Every question asked costs −1 reward, so the agent is incentivized to be strategic: ask the most likely disqualifier first, and exit early the moment it finds one.

This creates a genuine exploration-exploitation tradeoff that makes for a rich RL training signal.

---

## How It Works

┌─────────────────────────────────────────────────────────┐
│ RL Agent │
│ "Which field should I ask to disqualify fastest?" │
└──────────────────────┬──────────────────────────────────┘
│ action: ask("egfr") or decide(eligible=False)
▼
┌─────────────────────────────────────────────────────────┐
│ ClinicalTrialEnv │
│ Hidden patient profile. Reveals one field per step. │
│ Grades final decision against ground truth criteria. │
└──────────────────────┬──────────────────────────────────┘
│ observation: {revealed_fields, trial_criteria, reward}
▼
┌─────────────────────────────────────────────────────────┐
│ Reward Signal │
│ −1 per question · +20 correct · −20 wrong · +2 reason │
└─────────────────────────────────────────────────────────┘

---

## Action Space

Two action types:

**Ask** — request one patient field:

```json
{
  "action_type": "ask",
  "field_request": "egfr"
}
```

Valid field names: `age`, `egfr`, `hba1c`, `medications`, `conditions`

**Decide** — submit final eligibility ruling:

```json
{
  "action_type": "decide",
  "eligible": false,
  "reason": "eGFR 38 is below the minimum threshold of 45"
}
```

---

## Observation Space

| Field             | Type | Description                        |
| ----------------- | ---- | ---------------------------------- |
| `revealed_fields` | dict | Patient fields revealed so far     |
| `trial_criteria`  | dict | Eligibility rules (always visible) |
| `questions_asked` | int  | Running count of questions used    |
| `last_answer`     | str  | Answer to the last question asked  |
| `task_id`         | str  | Which task is running              |
| `decision_made`   | bool | Whether episode has ended          |

---

## Tasks

### single_criterion — Easy

One rule: patient age must be between 18 and 65.
Optimal strategy: ask age, decide immediately. One question maximum.

### multi_criteria — Medium

Four rules: age 30–70, eGFR ≥ 45, HbA1c ≤ 8.0, no warfarin or insulin.
Partial credit per criterion. Optimal strategy: ask most likely disqualifier first.

### edge_case — Hard

Multiple criteria with borderline values. eGFR may be exactly 44, 45, or 46.
HbA1c range is 6.5–9.5. Agent must reason carefully about threshold edge cases.

---

## Reward Structure

| Event                         | Reward           | Rationale                                    |
| ----------------------------- | ---------------- | -------------------------------------------- |
| Each question asked           | −1               | Incentivizes efficient information gathering |
| Correct final decision        | +20              | Primary objective signal                     |
| Wrong final decision          | −20              | Symmetric penalty                            |
| Reason cites specific value   | +2               | Rewards interpretable, grounded decisions    |
| Hitting max steps (10)        | −5               | Penalizes indecision                         |
| Partial criteria met (medium) | +1 per criterion | Encourages partial progress                  |

Final score normalized to (0, 1).

---

## Agent Strategy

The included agent in `inference.py` uses chain-of-thought reasoning via an LLM to decide which field to ask next:

1. Given revealed fields and trial criteria, reason about which unrevealed field is most likely to be a disqualifier
2. Ask that field
3. If a disqualifier is found, exit early — no need to ask more questions
4. If all relevant fields pass, decide eligible with a cited reason for bonus reward

This adaptive strategy significantly outperforms naive fixed-order questioning.

---

## Baseline Scores

| Task             | Avg Questions | Avg Score |
| ---------------- | ------------- | --------- |
| single_criterion | 1.0           | 0.93      |
| multi_criteria   | 2.3           | 0.81      |
| edge_case        | 2.8           | 0.76      |

Scores vary by patient because criteria thresholds and patient values are randomized each episode.

---

## Using for RL Training

ClinicalTrialEnv is designed to plug directly into RL training frameworks like TRL or GRPO:

```python
from openenv.core import EnvClient
from client import ClinicalTrialEnv

# Connect to the live environment
with ClinicalTrialEnv(
    base_url="https://mohonhf-clinical-trial-env.hf.space"
).sync() as env:

    # Run a full episode
    result = env.reset(task_id="multi_criteria")

    while not result.done:
        # Your policy generates an action
        action = your_policy(result.observation)
        result = env.step(action)

    # Use reward signal for GRPO/PPO training
    episode_reward = result.reward
```

Compatible with TRL, Oumi, SkyRL, and any OpenEnv-compatible training framework.

## Setup

```bash
pip install openenv-core
cd clinical_trial_env
uv run inference.py
```

Environment variables:

```bash
HF_TOKEN=hf_your_token_here           # required
API_BASE_URL=https://router.huggingface.co/v1  # default provided
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct           # default provided
```

---

## Project Structure

clinical_trial_env/
├── inference.py # LLM agent with chain-of-thought field selection
├── models.py # ClinicalTrialAction, ClinicalTrialObservation
├── tasks.py # 3 task definitions + graders
├── data.py # Patient profile generators
├── client.py # WebSocket client wrapper
├── openenv.yaml # Task registry metadata
├── pyproject.toml
└── server/
├── app.py # FastAPI server + /web frontend
├── clinical_trial_env_environment.py # Core environment logic
└── static/
└── index.html # Live interactive demo

---

## Live Demo

Visit the interactive demo at **[https://mohonhf-clinical-trial-env.hf.space/web](https://mohonhf-clinical-trial-env.hf.space/web)**

Watch the agent interview a real patient in real time via WebSocket, with live reward tracking and criteria checking.

---

Built with FastAPI · Docker · OpenEnv · Qwen2.5-72B · Meta PyTorch OpenEnv Hackathon 2026

import os
from openai import OpenAI
from server.clinical_trial_env_environment import ClinicalTrialEnvironment
from models import ClinicalTrialAction

# ===== ENV VARIABLES =====
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")
API_KEY      = os.getenv("API_KEY", HF_TOKEN)

if API_KEY is None:
    raise ValueError("API_KEY or HF_TOKEN environment variable is required")

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

BENCHMARK = "clinical_trial"
MAX_STEPS = 10

TASK_FIELDS = {
    "single_criterion": ["age"],
    "multi_criteria":   ["egfr", "age", "hba1c", "medications"],
    "edge_case":        ["egfr", "hba1c", "age", "medications", "conditions"],
}


# ===== NORMALIZE =====
def normalize_score(r):
    score = (r + 20) / 42
    return max(0.001, min(0.999, score))


# ===== LOG FUNCTIONS =====
def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error=None):
    done_str = str(done).lower()
    error_val = error if error else "null"
    normalized = normalize_score(reward)
    print(
        f"[STEP] step={step} action={action} reward={normalized:.2f} "
        f"done={done_str} error={error_val}",
        flush=True
    )

def log_end(success: bool, steps: int, rewards: list):
    normalized = [normalize_score(r) for r in rewards]
    rewards_str = ",".join(f"{r:.2f}" for r in normalized)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True
    )


# ===== LLM DECISION =====
def get_decision_from_llm(revealed: dict, criteria: dict) -> tuple[bool, str]:
    """Use LLM via proxy to decide eligibility based on revealed fields."""
    prompt = f"""You are a clinical trial eligibility screener.

Trial criteria:
{criteria}

Revealed patient fields:
{revealed}

Based only on the revealed fields and the trial criteria, is this patient eligible?
Respond in exactly this format:
ELIGIBLE: true or false
REASON: one sentence explanation referencing specific values
"""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
    )
    text = response.choices[0].message.content.strip()
    eligible = "eligible: true" in text.lower()
    reason = text.split("REASON:")[-1].strip() if "REASON:" in text else text
    return eligible, reason


# ===== SINGLE TASK EPISODE =====
def run_task(task_id: str):
    env = ClinicalTrialEnvironment()
    step_result = env.reset(task_id=task_id)

    obs = getattr(step_result, "observation", step_result)

    rewards = []
    step_count = 0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    fields_to_ask = TASK_FIELDS.get(task_id, ["age"])

    try:
        for field in fields_to_ask:
            if step_count >= MAX_STEPS - 1:
                break

            action = ClinicalTrialAction(action_type="ask", field_request=field)
            step_result = env.step(action)

            reward = float(getattr(step_result, "reward", 0.0) or 0.0)
            done = bool(getattr(step_result, "done", False))
            obs = getattr(step_result, "observation", step_result)

            step_count += 1
            rewards.append(reward)
            log_step(step_count, f"ask:{field}", reward, done)

            if done:
                success = reward >= 0.5
                break

        else:
            # Asked all fields — use LLM for final decision
            eligible, reason = get_decision_from_llm(
                obs.revealed_fields, obs.trial_criteria
            )
            action = ClinicalTrialAction(
                action_type="decide",
                eligible=eligible,
                reason=reason
            )
            step_result = env.step(action)
            reward = float(getattr(step_result, "reward", 0.0) or 0.0)
            done = bool(getattr(step_result, "done", False))

            step_count += 1
            rewards.append(reward)
            log_step(
                step_count,
                f"decide:{'eligible' if eligible else 'not_eligible'}",
                reward,
                done
            )
            success = reward >= 0.5

    finally:
        log_end(success=success, steps=step_count, rewards=rewards)


# ===== RUN ALL 3 TASKS =====
if __name__ == "__main__":
    for task_id in ["single_criterion", "multi_criteria", "edge_case"]:
        run_task(task_id)
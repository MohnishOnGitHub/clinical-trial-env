import os
from openai import OpenAI
from server.clinical_trial_env_environment import ClinicalTrialEnvironment
from models import ClinicalTrialAction  # ✅ fixed import

# ===== ENV VARIABLES =====
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

BENCHMARK = "clinical_trial"
MAX_STEPS = 10

TASK_FIELDS = {
    "single_criterion": ["age"],
    "multi_criteria":   ["egfr", "age", "hba1c", "medications"],
    "edge_case":        ["egfr", "hba1c", "age", "medications", "conditions"],
}


# ===== LOG FUNCTIONS =====
def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error=None):
    done_str = str(done).lower()
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={done_str} error={error_val}",
        flush=True
    )

def log_end(success: bool, steps: int, rewards: list):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True
    )


# ===== BASELINE AGENT LOGIC =====
def get_decision(revealed: dict, criteria: dict) -> tuple[bool, str]:
    """Rule-based agent — checks each revealed field against criteria."""
    for field, value in revealed.items():
        if field == "age":
            age_min = criteria.get("age_min", 0)
            age_max = criteria.get("age_max", 999)
            if not (age_min <= value <= age_max):
                return False, f"age {value} outside range {age_min}-{age_max}"
        if field == "egfr":
            egfr_min = criteria.get("egfr_min", 0)
            if value is None or value < egfr_min:  # ✅ handles None (your earlier bug)
                return False, f"egfr {value} below minimum {egfr_min}"
        if field == "hba1c":
            hba1c_max = criteria.get("hba1c_max", 999)
            if value is not None and value > hba1c_max:
                return False, f"hba1c {value} above maximum {hba1c_max}"
        if field == "medications":
            excluded = criteria.get("excluded_meds", [])
            bad = [m for m in (value or []) if m in excluded]
            if bad:
                return False, f"excluded medications present: {bad}"
    return True, "all revealed criteria met"


# ===== SINGLE TASK EPISODE =====
def run_task(task_id: str):
    env = ClinicalTrialEnvironment()
    step_result = env.reset(task_id=task_id)

    # ✅ Handle both flat and nested reset returns
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

            action = ClinicalTrialAction(action_type="ask", field_request=field)  # ✅
            step_result = env.step(action)

            # ✅ Safely unpack result
            reward = float(getattr(step_result, "reward", 0.0) or 0.0)
            done = bool(getattr(step_result, "done", False))
            obs = getattr(step_result, "observation", step_result)

            step_count += 1
            rewards.append(reward)
            log_step(step_count, f"ask:{field}", reward, done)

            if done:
                success = reward >= 0.5
                break

            # Check if we already know it fails — decide early
            early_eligible, early_reason = get_decision(
                obs.revealed_fields, obs.trial_criteria
            )
            if not early_eligible:
                action = ClinicalTrialAction(  # ✅
                    action_type="decide",
                    eligible=False,
                    reason=early_reason
                )
                step_result = env.step(action)
                reward = float(getattr(step_result, "reward", 0.0) or 0.0)
                done = bool(getattr(step_result, "done", False))

                step_count += 1
                rewards.append(reward)
                log_step(step_count, "decide:not_eligible", reward, done)
                success = reward >= 0.5
                break

        else:
            # Asked all fields — make final decision
            eligible, reason = get_decision(obs.revealed_fields, obs.trial_criteria)
            action = ClinicalTrialAction(  # ✅
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
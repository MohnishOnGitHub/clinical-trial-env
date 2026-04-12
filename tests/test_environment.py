import pytest
import random
from models import ClinicalTrialAction, ClinicalTrialObservation
from server.clinical_trial_env_environment import ClinicalTrialEnvironment
from tasks import EasyTask, MediumTask, HardTask, TASKS


# ===== MODEL TESTS =====

def test_action_ask_valid():
    action = ClinicalTrialAction(action_type="ask", field_request="age")
    assert action.action_type == "ask"
    assert action.field_request == "age"


def test_action_decide_valid():
    action = ClinicalTrialAction(action_type="decide", eligible=True, reason="age 45 within range")
    assert action.eligible == True
    assert action.reason is not None


def test_observation_defaults():
    obs = ClinicalTrialObservation()
    assert obs.revealed_fields == {}
    assert obs.questions_asked == 0
    assert obs.decision_made == False
    assert obs.done == False
    assert 0.0 < obs.reward < 1.0


# ===== TASK TESTS =====

def test_easy_task_criteria():
    task = EasyTask()
    criteria = task.get_criteria()
    assert "age_min" in criteria
    assert "age_max" in criteria
    assert criteria["age_min"] == 18
    assert criteria["age_max"] == 65


def test_easy_task_patient_generation():
    task = EasyTask()
    for _ in range(20):
        patient = task.generate_patient()
        assert "age" in patient
        assert 10 <= patient["age"] <= 80


def test_medium_task_criteria():
    task = MediumTask()
    criteria = task.get_criteria()
    assert "egfr_min" in criteria
    assert "hba1c_max" in criteria
    assert "no_meds" in criteria


def test_hard_task_borderline_values():
    task = HardTask()
    egfr_values = set()
    for _ in range(50):
        patient = task.generate_patient()
        egfr_values.add(patient["egfr"])
    assert egfr_values.issubset({44, 45, 46})


def test_task_registry():
    assert "single_criterion" in TASKS
    assert "multi_criteria" in TASKS
    assert "edge_case" in TASKS


# ===== GRADER TESTS =====

def test_easy_grader_correct_eligible():
    task = EasyTask()
    patient = {"age": 45}
    action = ClinicalTrialAction(action_type="decide", eligible=True, reason="age 45 within range 18-65")
    raw = task.grade(action, patient, questions_asked=1)
    assert raw > 0


def test_easy_grader_correct_not_eligible():
    task = EasyTask()
    patient = {"age": 70}
    action = ClinicalTrialAction(action_type="decide", eligible=False, reason="age 70 outside range")
    raw = task.grade(action, patient, questions_asked=1)
    assert raw > 0


def test_easy_grader_wrong_decision():
    task = EasyTask()
    patient = {"age": 45}
    action = ClinicalTrialAction(action_type="decide", eligible=False, reason="wrong")
    raw = task.grade(action, patient, questions_asked=1)
    assert raw < 0


def test_grader_reason_bonus():
    task = EasyTask()
    patient = {"age": 45}
    action_with_reason = ClinicalTrialAction(action_type="decide", eligible=True, reason="age 45 within range 18-65")
    action_no_reason = ClinicalTrialAction(action_type="decide", eligible=True, reason="looks good")
    raw_with = task.grade(action_with_reason, patient, questions_asked=1)
    raw_without = task.grade(action_no_reason, patient, questions_asked=1)
    assert raw_with > raw_without


def test_grader_question_penalty():
    task = EasyTask()
    patient = {"age": 45}
    action = ClinicalTrialAction(action_type="decide", eligible=True, reason="age 45 ok")
    raw_few = task.grade(action, patient, questions_asked=1)
    raw_many = task.grade(action, patient, questions_asked=5)
    assert raw_few > raw_many


# ===== ENVIRONMENT TESTS =====

def test_env_reset():
    env = ClinicalTrialEnvironment()
    obs = env.reset(task_id="single_criterion")
    assert obs is not None
    assert obs.revealed_fields == {}
    assert obs.questions_asked == 0
    assert obs.decision_made == False
    assert obs.trial_criteria != {}


def test_env_reset_all_tasks():
    env = ClinicalTrialEnvironment()
    for task_id in ["single_criterion", "multi_criteria", "edge_case"]:
        obs = env.reset(task_id=task_id)
        assert obs.task_id == task_id
        assert obs.trial_criteria != {}


def test_env_ask_valid_field():
    env = ClinicalTrialEnvironment()
    env.reset(task_id="single_criterion")
    action = ClinicalTrialAction(action_type="ask", field_request="age")
    obs = env.step(action)
    assert "age" in obs.revealed_fields
    assert obs.questions_asked == 1
    assert obs.done == False


def test_env_ask_invalid_field():
    env = ClinicalTrialEnvironment()
    env.reset(task_id="single_criterion")
    action = ClinicalTrialAction(action_type="ask", field_request="nonexistent_field")
    obs = env.step(action)
    assert obs.last_answer == "invalid_field"


def test_env_ask_same_field_twice():
    env = ClinicalTrialEnvironment()
    env.reset(task_id="multi_criteria")
    action = ClinicalTrialAction(action_type="ask", field_request="age")
    env.step(action)
    obs = env.step(action)
    assert obs.questions_asked == 1


def test_env_decide_ends_episode():
    env = ClinicalTrialEnvironment()
    env.reset(task_id="single_criterion")
    action = ClinicalTrialAction(action_type="decide", eligible=True, reason="test")
    obs = env.step(action)
    assert obs.done == True
    assert obs.decision_made == True


def test_env_reward_in_range():
    env = ClinicalTrialEnvironment()
    for task_id in ["single_criterion", "multi_criteria", "edge_case"]:
        for eligible in [True, False]:
            env.reset(task_id=task_id)
            action = ClinicalTrialAction(action_type="decide", eligible=eligible, reason="age 45 egfr 50")
            obs = env.step(action)
            assert 0.0 < obs.reward < 1.0, f"Reward {obs.reward} out of range for {task_id}"


def test_env_max_steps():
    env = ClinicalTrialEnvironment()
    env.reset(task_id="multi_criteria")
    for _ in range(9):
        action = ClinicalTrialAction(action_type="ask", field_request="age")
        obs = env.step(action)
    assert obs.done == True


def test_env_state():
    env = ClinicalTrialEnvironment()
    env.reset(task_id="single_criterion")
    state = env.state
    assert state.step_count == 0
    env.step(ClinicalTrialAction(action_type="ask", field_request="age"))
    assert env.state.step_count == 1


def test_full_episode_easy():
    env = ClinicalTrialEnvironment()
    env.reset(task_id="single_criterion")
    env.step(ClinicalTrialAction(action_type="ask", field_request="age"))
    obs = env.step(ClinicalTrialAction(action_type="decide", eligible=True, reason="age within range"))
    assert obs.done == True
    assert 0.0 < obs.reward < 1.0


def test_full_episode_medium():
    env = ClinicalTrialEnvironment()
    env.reset(task_id="multi_criteria")
    for field in ["egfr", "age", "hba1c", "medications"]:
        obs = env.step(ClinicalTrialAction(action_type="ask", field_request=field))
        if obs.done:
            break
    if not obs.done:
        obs = env.step(ClinicalTrialAction(action_type="decide", eligible=True, reason="egfr 50 age 45 hba1c 7.0"))
    assert obs.done == True
    assert 0.0 < obs.reward < 1.0
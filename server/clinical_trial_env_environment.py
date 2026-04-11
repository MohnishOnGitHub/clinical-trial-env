from uuid import uuid4
import random
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from tasks import TASKS
except ImportError:
    from ..tasks import TASKS

from models import ClinicalTrialAction, ClinicalTrialObservation


def normalize(raw):
    score = (raw + 20) / 42
    result = max(0.001, min(0.999, score))
    return float(result)


class ClinicalTrialEnvironment(Environment):

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.hidden_patient = {}
        self.revealed = {}
        self.criteria = {}
        self.questions_asked = 0
        self.done = False
        self.task_id = "single_criterion"
        self.task = None

    def build_obs(self, reward, done, last_answer=None):
        if done:
            normalized_reward = normalize(reward)
        else:
            normalized_reward = 0.476
        normalized_reward = float(normalized_reward)
        normalized_reward = max(0.001, min(0.999, normalized_reward))
        return ClinicalTrialObservation(
            revealed_fields=self.revealed,
            last_answer=last_answer,
            trial_criteria=self.criteria,
            questions_asked=self.questions_asked,
            task_id=self.task_id,
            decision_made=done,
            done=done,
            reward=normalized_reward
        )

    def reset(self, task_id: str = None) -> ClinicalTrialObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)

        if task_id and task_id in TASKS:
            self.task = TASKS[task_id]
        else:
            self.task = random.choice(list(TASKS.values()))

        self.task_id = self.task.task_id
        self.hidden_patient = self.task.generate_patient()

        self.hidden_patient.setdefault("age", random.randint(10, 80))
        self.hidden_patient.setdefault("egfr", random.randint(20, 90))
        self.hidden_patient.setdefault("hba1c", round(random.uniform(5.0, 10.0), 1))
        self.hidden_patient.setdefault("medications", ["none"])
        self.hidden_patient.setdefault("conditions", [])

        self.criteria = self.task.get_criteria()
        self.revealed = {}
        self.questions_asked = 0
        self.done = False

        return self.build_obs(0.0, False)

    def step(self, action: ClinicalTrialAction) -> ClinicalTrialObservation:
        self._state.step_count += 1

        if self._state.step_count >= 10:
            self.done = True
            return self.build_obs(-5.0, True)

        if self.done:
            return self.build_obs(0.0, True)

        if action.action_type == "ask":
            field = action.field_request

            if field not in self.hidden_patient:
                return self.build_obs(-1.0, False, "invalid_field")

            if field in self.revealed:
                return self.build_obs(-0.5, False, str(self.revealed[field]))

            value = self.hidden_patient[field]
            self.revealed[field] = value
            self.questions_asked += 1

            return self.build_obs(-1.0, False, str(value))

        elif action.action_type == "decide":
            self.done = True
            reward = self.task.grade(
                action,
                self.hidden_patient,
                self.questions_asked
            )
            return self.build_obs(reward, True)

        return self.build_obs(-1.0, False, "unknown_action_type")

    @property
    def state(self) -> State:
        return State(
            episode_id=self._state.episode_id,
            step_count=self._state.step_count,
            metadata={
                "hidden_patient": self.hidden_patient,
                "revealed_fields": self.revealed,
                "criteria": self.criteria,
                "questions_asked": self.questions_asked,
                "done": self.done,
                "task_id": self.task_id
            }
        )
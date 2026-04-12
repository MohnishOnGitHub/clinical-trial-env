from openenv.core.env_server.types import Action, Observation
from typing import Literal, Optional, Dict, Union


class ClinicalTrialAction(Action):
    action_type: Literal['ask', 'decide']
    field_request: Optional[str] = None
    eligible: Optional[bool] = None
    reason: Optional[str] = None


class ClinicalTrialObservation(Observation):
    revealed_fields: Dict = {}
    last_answer: Optional[str] = None
    trial_criteria: Dict = {}
    questions_asked: int = 0
    task_id: str = ''
    decision_made: bool = False
    done: bool = False
    reward: float = 0.476
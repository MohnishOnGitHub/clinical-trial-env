"""Clinical Trial Env - Client wrapper."""

from typing import Dict
from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State
from .models import ClinicalTrialAction, ClinicalTrialObservation


class ClinicalTrialEnv(
    EnvClient[ClinicalTrialAction, ClinicalTrialObservation, State]
):
    """
    Client for the Clinical Trial Environment.

    Maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions.

    Example:
        >>> with ClinicalTrialEnv(base_url="http://localhost:7860") as client:
        ...     result = client.reset()
        ...     action = ClinicalTrialAction(action_type="ask", field_request="age")
        ...     result = client.step(action)
        ...     action = ClinicalTrialAction(action_type="decide", eligible=True, reason="age ok")
        ...     result = client.step(action)
    """

    def _step_payload(self, action: ClinicalTrialAction) -> Dict:
        payload = {"action_type": action.action_type}
        if action.action_type == "ask":
            payload["field_request"] = action.field_request
        elif action.action_type == "decide":
            payload["eligible"] = action.eligible
            payload["reason"] = action.reason
        return payload

    def _parse_result(self, payload: Dict) -> StepResult[ClinicalTrialObservation]:
        obs_data = payload.get("observation", {})
        observation = ClinicalTrialObservation(
            revealed_fields=obs_data.get("revealed_fields", {}),
            last_answer=obs_data.get("last_answer"),
            trial_criteria=obs_data.get("trial_criteria", {}),
            questions_asked=obs_data.get("questions_asked", 0),
            task_id=obs_data.get("task_id", ""),
            decision_made=obs_data.get("decision_made", False),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
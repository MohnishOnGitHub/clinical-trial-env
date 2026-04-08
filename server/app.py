from fastapi import FastAPI
from fastapi.responses import JSONResponse
from openenv.core.env_server.http_server import create_app

from models import ClinicalTrialAction, ClinicalTrialObservation
from server.clinical_trial_env_environment import ClinicalTrialEnvironment

# Create OpenEnv app
openenv_app = create_app(
    ClinicalTrialEnvironment,
    ClinicalTrialAction,
    ClinicalTrialObservation,
    env_name="clinical_trial_env",
    max_concurrent_envs=1,
)

# Mount OpenEnv at root so /reset, /step, /close are all reachable
app = openenv_app

# Add health routes on top
@app.get("/")
def root():
    return JSONResponse({"status": "ok", "space": "clinical_trial_env"})

@app.get("/health")
def health():
    return JSONResponse({"status": "ok"})
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from openenv.core.env_server.http_server import create_app

from models import ClinicalTrialAction, ClinicalTrialObservation
from server.clinical_trial_env_environment import ClinicalTrialEnvironment


# Create OpenEnv app FIRST
openenv_app = create_app(
    ClinicalTrialEnvironment,
    ClinicalTrialAction,
    ClinicalTrialObservation,
    env_name="clinical_trial_env",
    max_concurrent_envs=1,
)

# Now wrap it in FastAPI
app = FastAPI()

# Health routes MUST come before mount
@app.get("/")
def root():
    return JSONResponse({"status": "ok", "space": "clinical_trial_env"})

@app.get("/health")
def health():
    return JSONResponse({"status": "ok"})

@app.get("/ready")
def ready():
    return JSONResponse({"status": "ready"})

# Mount OpenEnv at /env (NOT / — that would swallow the health routes)
app.mount("/env", openenv_app)
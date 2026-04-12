from fastapi import FastAPI
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from openenv.core.env_server.http_server import create_app
import os

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

# Serve the frontend HTML
@app.get("/web", response_class=HTMLResponse)
def frontend():
    static_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    with open(static_path, "r") as f:
        return HTMLResponse(content=f.read())

# Add health routes on top
@app.get("/")
def root():
    return JSONResponse({"status": "ok", "space": "clinical_trial_env"})

@app.get("/health")
def health():
    return JSONResponse({"status": "ok"})

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
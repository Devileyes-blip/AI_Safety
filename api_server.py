from fastapi import FastAPI
from pydantic import BaseModel

from risk_checker import analyze_prompt

app = FastAPI(title="Prompt Injection Detector API")


class PromptRequest(BaseModel):
    prompt: str


@app.post("/check_prompt")
def check_prompt(request: PromptRequest):

    result = analyze_prompt(request.prompt)

    return {
        "prompt": request.prompt,
        "risk_level": result["risk_level"],
        "risk_score": result["final_score"],
        "action": result["action"],
        "attack_type": result["attack_type"]
    }
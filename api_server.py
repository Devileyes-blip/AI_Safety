from fastapi import FastAPI, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from risk_checker import analyze_prompt

app = FastAPI(title="Prompt Injection Detector API")


class PromptRequest(BaseModel):
    prompt: str


@app.post("/check_prompt")
def check_prompt(request: PromptRequest):
    analysis = analyze_prompt(request.prompt)

    response_body = {
        "prompt": request.prompt,
        "risk_level": analysis["risk_level"],
        "risk_score": analysis["final_score"],
        "action": analysis["action"],
        "attack_type": analysis["attack_type"]
    }

    if analysis["action"] == "BLOCK":
        response_body["status"] = "blocked"
        response_body["message"] = "Prompt blocked due to high prompt-injection risk."
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content=response_body
        )

    if analysis["action"] == "REVIEW":
        response_body["status"] = "review"
        response_body["message"] = "Prompt flagged for manual review before processing."
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content=response_body
        )

    response_body["status"] = "allowed"
    response_body["message"] = "Prompt is safe to process."
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=response_body
    )

from pydantic import BaseModel, Field
from typing import List


class Claim(BaseModel):
    subject: str
    predicate: str
    object: str
    confidence: int


class Claims(BaseModel):
    claims: List[Claim] = Field(..., min_length=5)


class ClaimTool:
    tools = [{
        "type": "function",
        "function": {
            "name": "extractClaim",
            "parameters": Claims.model_json_schema()
        }
    }]
    tool_choice = {"type": "function", "function": {"name": "extractClaim"}}

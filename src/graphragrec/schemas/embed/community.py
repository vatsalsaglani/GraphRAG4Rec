from typing import List, Dict, Union
from pydantic import BaseModel, Field


class Finding(BaseModel):
    summary: str
    explanation: str


class CommunityReport(BaseModel):
    title: str
    summary: str
    rating: int
    rating_explanation: str
    findings: List[Finding] = Field(..., min_length=8, max_length=15)


class CommunityReportTool:
    tools = [{
        "type": "function",
        "function": {
            "name": "createCommunityReport",
            "parameters": CommunityReport.model_json_schema()
        }
    }]
    tool_choice = {
        "type": "function",
        "function": {
            "name": "createCommunityReport"
        }
    }

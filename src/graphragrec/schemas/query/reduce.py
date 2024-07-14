from pydantic import BaseModel, Field
from typing import List, Optional


class RecommendedCommunity(BaseModel):
    community_id: int
    relevance_score: int = Field(..., ge=1, le=5)
    relevance_explanation: str
    key_movies: List[str]
    key_themes: List[str]


class ReducerOutput(BaseModel):
    map_output_synthesis: str
    contradiction_resolution: Optional[str]
    recommended_communities: List[RecommendedCommunity]
    overall_themes: List[str]
    query_relation_explanation: str
    additional_considerations: Optional[str]


class ReduceTool:
    tools = [{
        "type": "function",
        "function": {
            "name": "reduceOutupt",
            "parameters": ReducerOutput.model_json_schema()
        }
    }]
    tool_choice = {"type": "function", "function": {"name": "reduceOutupt"}}

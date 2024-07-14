from pydantic import BaseModel, Field
from typing import List, Dict, Optional


class SubQuery(BaseModel):
    query: str
    explanation: str


class KeyFactor(BaseModel):
    factor_type: str = Field(
        ...,
        description=
        "Key factor can be 'keyword', 'actor', 'director', 'star', etc.")
    value: str
    explanation: str


class CommunityThought(BaseModel):
    community_id: int
    relevance: bool
    explanation: str


class SubQueryAnalysis(BaseModel):
    relevant_movie_types: str
    irrelevant_movie_types: str
    key_factors: List[KeyFactor]
    relevant_communities: List[CommunityThought]
    irrelevant_communities: List[CommunityThought]


class RelevantCommunity(BaseModel):
    community_id: int
    relevance_score: int = Field(..., ge=1, le=5)
    relevance_reason: str
    key_movies: List[str]
    key_themes: List[str]


class MapOutput(BaseModel):
    query_analysis: str
    query_decomposition: List[SubQuery]
    sub_query_analysis: Dict[str, SubQueryAnalysis]
    final_relevant_communities: List[RelevantCommunity]
    irrelevant_communities: List[str]


class MapTool:
    tools = [{
        "type": "function",
        "function": {
            "name": "mapOutupt",
            "parameters": MapOutput.model_json_schema()
        }
    }]
    tool_choice = {"type": "function", "function": {"name": "mapOutupt"}}

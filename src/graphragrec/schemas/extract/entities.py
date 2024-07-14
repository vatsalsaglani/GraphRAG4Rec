from pydantic import BaseModel, Field
from typing import List


class Entity(BaseModel):
    name: str
    type: str
    description: str


class Entities(BaseModel):
    entities: List[Entity] = Field(..., min_length=5)


class EntityTool:
    tools = [{
        "type": "function",
        "function": {
            "name": "extractEntities",
            "parameters": Entities.model_json_schema()
        }
    }]
    tool_choice = {"type": "function", "function": {"name": "extractEntities"}}

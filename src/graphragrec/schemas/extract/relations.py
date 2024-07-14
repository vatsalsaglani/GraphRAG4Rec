from pydantic import BaseModel, Field
from typing import List


class Relation(BaseModel):
    source: str
    target: str
    type: str
    description: str
    strength: int


class Relations(BaseModel):
    relations: List[Relation] = Field(..., min_length=5)


class RelationTool:
    tools = [{
        "type": "function",
        "function": {
            "name": "extractRelations",
            "parameters": Relations.model_json_schema()
        }
    }]
    tool_choice = {
        "type": "function",
        "function": {
            "name": "extractRelations"
        }
    }

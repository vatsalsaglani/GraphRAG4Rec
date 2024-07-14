'''Community Report from multiple Community Summary'''
import random
import asyncio
import json_repair as json
from typing import List, Dict
from llm.localllm import LocalLLM
from graphragrec.prompts.prompts import EMBED
from graphragrec.utils.usage import calculateUsages
from graphragrec.schemas.embed.community import CommunityReportTool


async def combineCommunityReports(llm: LocalLLM, model: str,
                                  community_sumary: List[Dict]):
    message_content = f"```{community_sumary}```"
    messages = [{
        "role": "system",
        "content": EMBED.COMBINE
    }, {
        "role": "user",
        "content": message_content
    }]
    output, usage = await llm.__function_call__(
        messages,
        model,
        CommunityReportTool.tools,
        tool_choice=CommunityReportTool.tool_choice)
    return output, usage

import random
import asyncio
import time
import logging
from typing import List, Dict
from tqdm.auto import trange
from llm.localllm import LocalLLM
from graphragrec.prompts.prompts import QUERY
from graphragrec.utils.usage import calculateUsages
from graphragrec.schemas.query.map import MapTool


async def queryMap(llm: LocalLLM,
                   query: str,
                   batched_community_reports: List[Dict],
                   max_allowed_concurrency: int = 2,
                   default_model: str = "gpt-4o"):
    mapOutputs = []
    usages = []

    async def processMap(query: str, report_batch: Dict):
        message_content = f'Query: `{query}`\n Batch Community Reports: ```{report_batch}```'
        messages = [{
            "role": "system",
            "content": QUERY.MAP
        }, {
            "role": "user",
            "content": message_content
        }]
        time.sleep(random.choice([0.5, 1]))
        output, usage = await llm.__function_call__(
            messages,
            default_model,
            MapTool.tools,
            tool_choice=MapTool.tool_choice,
            max_length=4095)
        mapOutputs.append(output)
        usages.append(usage)

    for ix in trange(0, len(batched_community_reports),
                     max_allowed_concurrency):
        intermediate_batches = batched_community_reports[
            ix:ix + max_allowed_concurrency]
        await asyncio.gather(
            *[processMap(query, b) for b in intermediate_batches])

    usages = calculateUsages(usages)
    return mapOutputs, usages


if __name__ == "__main__":
    import json
    import asyncio
    from configs import OPENAI_API_KEY
    batched_communities = json.loads(
        open("./output/v7-all/batched-community-reports.json").read())
    user_query = "I want to have a fun weekend movie session with my bros."
    llm = LocalLLM(api_key=OPENAI_API_KEY)
    output, usage = asyncio.run(queryMap(llm, user_query, batched_communities))
    print(f'USAGE: {usage}')
    print(f"OUTPUT: \n {json.dumps(output, indent=4)}")

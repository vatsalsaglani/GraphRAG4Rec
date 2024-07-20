from typing import List, Dict
import json_repair as json
from llm.localllm import LocalLLM
from graphragrec.prompts.prompts import EXTRACT
from graphragrec.utils.usage import calculateUsages
from graphragrec.schemas.extract.relations import RelationTool

ALLOWED_GLEANINGS = 1


async def extractRelations(llm: LocalLLM, model: str, entities: List[Dict]):
    relations = []
    usages = []
    for gleaning in range(ALLOWED_GLEANINGS):
        message_content = f"```{entities}```"
        messages = [{
            "role": "system",
            "content": EXTRACT.RELATION
        }, {
            "role":
            "user",
            "content":
            message_content if gleaning == 0 else
            f'The following are the relations you detected: {relations}. You have missed some relations, please provide all the relations. {message_content}'
        }]
        # output, usage = await llm.__complete__(messages, model)
        output, usage = await llm.__function_call__(
            messages,
            model,
            RelationTool.tools,
            tool_choice=RelationTool.tool_choice)
        relations_ = output
        relations += relations_.get("relations")
        usages += [usage]
    usages = calculateUsages(usages)
    return relations, usages


if __name__ == "__main__":
    import asyncio
    from configs import OPENAI_API_KEY
    llm = LocalLLM(api_key=OPENAI_API_KEY)
    output = asyncio.run(
        extractRelations(llm, "gpt-4o-mini", [{
            'name':
            'The Shawshank Redemption',
            'type':
            'Movie',
            'id':
            '1',
            'description':
            'A drama film about two imprisoned men who develop a deep bond.'
        }, {
            'name':
            'Frank Darabont',
            'type':
            'Director',
            'id':
            '2',
            'description':
            'Director of The Shawshank Redemption.'
        }, {
            'name':
            'Tim Robbins',
            'type':
            'Actor',
            'id':
            '3',
            'description':
            'Actor in The Shawshank Redemption.'
        }, {
            'name':
            'Morgan Freeman',
            'type':
            'Actor',
            'id':
            '4',
            'description':
            'Actor in The Shawshank Redemption.'
        }, {
            'name':
            'Bob Gunton',
            'type':
            'Actor',
            'id':
            '5',
            'description':
            'Actor in The Shawshank Redemption.'
        }, {
            'name':
            'William Sadler',
            'type':
            'Actor',
            'id':
            '6',
            'description':
            'Actor in The Shawshank Redemption.'
        }, {
            'name':
            'The Shawshank Redemption',
            'type':
            'Movie',
            'id':
            '1',
            'description':
            'A drama film about two imprisoned men who develop a deep bond and find solace and eventual redemption through acts of common decency.'
        }, {
            'name':
            'Frank Darabont',
            'type':
            'Director',
            'id':
            '2',
            'description':
            'Director of The Shawshank Redemption.'
        }, {
            'name':
            'Tim Robbins',
            'type':
            'Actor',
            'id':
            '3',
            'description':
            'Actor in The Shawshank Redemption.'
        }, {
            'name':
            'Morgan Freeman',
            'type':
            'Actor',
            'id':
            '4',
            'description':
            'Actor in The Shawshank Redemption.'
        }, {
            'name':
            'Bob Gunton',
            'type':
            'Actor',
            'id':
            '5',
            'description':
            'Actor in The Shawshank Redemption.'
        }, {
            'name':
            'William Sadler',
            'type':
            'Actor',
            'id':
            '6',
            'description':
            'Actor in The Shawshank Redemption.'
        }, {
            'name':
            '1994',
            'type':
            'Year',
            'id':
            '7',
            'description':
            'The year The Shawshank Redemption was released.'
        }, {
            'name':
            'A',
            'type':
            'Certificate',
            'id':
            '8',
            'description':
            'The MPAA rating for The Shawshank Redemption.'
        }]))
    print(output)

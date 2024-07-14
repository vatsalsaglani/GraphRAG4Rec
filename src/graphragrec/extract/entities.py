import json_repair as json
from llm.localllm import LocalLLM
from graphragrec.prompts.prompts import EXTRACT
from graphragrec.utils.usage import calculateUsages
from graphragrec.schemas.extract.entities import EntityTool

ALLOWED_GLEANINGS = 1


async def extractEntities(llm: LocalLLM, model: str, content: str):
    entities = []
    usages = []
    for gleaning in range(ALLOWED_GLEANINGS):
        message_content = f"```{content}```"
        messages = [{
            "role": "system",
            "content": EXTRACT.ENTITIES
        }, {
            "role":
            "user",
            "content":
            message_content if gleaning == 0 else
            f'The following are the entities you provided: {entities}. You have missed some entities, please provide all the entities. {message_content}'
        }]
        # output, usage = await llm.__complete__(messages, model)
        output, usage = await llm.__function_call__(
            messages,
            model,
            EntityTool.tools,
            tool_choice=EntityTool.tool_choice)
        entities_ = output.get("entities")
        entities += entities_
        usages += [usage]
    usages = calculateUsages(usages)
    return entities, usages


if __name__ == "__main__":
    import asyncio
    from configs import OPENAI_API_KEY
    llm = LocalLLM(api_key=OPENAI_API_KEY)
    output = asyncio.run(
        extractEntities(
            llm, "gpt-4o",
            "Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.\nYear: 1994\nDirector: Frank Darabont\nCast: ['Tim Robbins', 'Morgan Freeman', 'Bob Gunton', 'William Sadler']\nCertificate: A"
        ))
    print(output)

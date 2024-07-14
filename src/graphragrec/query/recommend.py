import json
from typing import List, Dict
from llm.localllm import LocalLLM
from graphragrec.prompts.prompts import QUERY
from graphragrec.query.reports.map import queryMap
from graphragrec.query.reports.reduce import queryReduce
from graphragrec.utils.usage import calculateUsages

flatten = lambda lst: [item for sublist in lst for item in sublist]


async def recommend(llm: LocalLLM,
                    query: str,
                    batched_community_reports: List[Dict],
                    default_model: str = "gpt-4o"):
    usages = []
    yield "- Creating Search Keywords\n"
    mapOutputs, usage = await queryMap(llm,
                                       query,
                                       batched_community_reports,
                                       max_allowed_concurrency=4,
                                       default_model=default_model)
    usages += [usage]
    queries = list(
        set(
            flatten(
                list(
                    map(
                        lambda m: [
                            m.get("query").lower()
                            for m in m.get("query_decomposition")
                        ], mapOutputs)))))
    t = "\n".join(f'- *{q}*' for q in queries)
    yield "#### Keywords\n"
    yield t
    yield "\n"
    reducedOutputs, usage = await queryReduce(llm,
                                              query,
                                              mapOutputs,
                                              default_model=default_model)
    usages += [usage]
    yield "### Recommendation \n\n"
    user_message_content = f'Query: `{query}`\n Reducer Output: ```{reducedOutputs}```'
    messages = [{
        "role": "system",
        "content": QUERY.COMMUNICATE
    }, {
        "role": "user",
        "content": user_message_content
    }]
    async for chunk in llm.__stream__(messages, default_model,
                                      max_tokens=2000):
        yield chunk

    yield "\n\n\n TOKEN USAGE (MAP & REDUCE) \n\n"
    total = calculateUsages(usages)
    yield f"```json\n{json.dumps(total, indent=4)}\n```"


if __name__ == "__main__":
    import json
    import asyncio
    from configs import OPENAI_API_KEY
    batched_communities = json.loads(
        open("./output/v7-all/batched-community-reports.json").read())
    llm = LocalLLM(api_key=OPENAI_API_KEY)

    async def main(user_query: str):
        async for chunk in recommend(llm, user_query, batched_communities):
            print(chunk, end="", flush=True)

    while True:
        try:
            q = str(input("Enter Query: "))
            q = q.strip()
            if len(q) > 3:
                print(f'USER QUERY: {q}')
                asyncio.run(main(q))
                q = ""
        except KeyboardInterrupt:
            exit()

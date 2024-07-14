import json
import asyncio
from typing import AsyncGenerator
import streamlit as st
from configs import OPENAI_API_KEY
from llm.localllm import LocalLLM
from graphragrec.query.recommend import recommend

batched_communities = json.loads(
    open("./output/v7-all/batched-community-reports.json").read())
llm = LocalLLM(api_key=OPENAI_API_KEY)

st.set_page_config(page_title="GraphRAG4Rec", page_icon="🍿")


async def stream_async_generator(async_gen):
    async for item in async_gen:
        yield item


def main():
    st.title("GraphRAG4Recommendation: Movie recommendation using GraphRAG")
    user_query = st.text_input("What do you feel like watching today?",
                               key="user_query")
    if st.button("Recommend"):
        user_query = user_query.strip()
        if len(user_query.split(" ")) > 4:
            recommendation_generator = recommend(llm, user_query,
                                                 batched_communities)

            # Create a placeholder for the streaming output
            output_placeholder = st.empty()

            # Use asyncio to run the async generator
            async def run_async():
                full_output = ""
                async for chunk in stream_async_generator(
                        recommendation_generator):
                    full_output += chunk
                    output_placeholder.markdown(full_output)

            # Run the async function
            asyncio.run(run_async())
        else:
            st.error("Please provide a description longer than 4 words.")


if __name__ == "__main__":
    main()

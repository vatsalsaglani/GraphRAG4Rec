import re
# import json
# import logging
import asyncio
import random
import json_repair as json
import traceback
import backoff
from typing import Dict, List, Union
from openai import (AsyncOpenAI, RateLimitError, APITimeoutError,
                    InternalServerError, APIConnectionError)
from llm.base import BaseLLM


class LocalLLM(BaseLLM):

    def __init__(self, **kwargs):
        self.client = AsyncOpenAI(api_key=kwargs.get("api_key"))

    @backoff.on_exception(backoff.expo, RateLimitError, max_tries=8)
    async def __complete__(self, messages: List[Dict], model: str, **kwargs):
        managed_messages = messages
        await asyncio.sleep(
            random.choice([0.4, 0.5, 0.6, 0.9, 1, 0.3, 0.2, 0.1]))
        output = await self.client.chat.completions.create(
            messages=managed_messages, model=model, **kwargs)
        # print("OUTPUT: ", output)
        usage = output.usage.__dict__
        output_content = output.choices[0].message.content
        # print(f"OUTPUT CONTENT: {output_content}")
        return output_content, usage

    async def __stream__(self, messages: List[Dict], model: str, **kwargs):
        managed_messages = messages
        stream = await self.client.chat.completions.create(
            model=model, messages=managed_messages, stream=True, **kwargs)
        async for chunk in stream:
            yield chunk.choices[0].delta.content or ""

    @backoff.on_exception(backoff.expo, RateLimitError, max_tries=8)
    async def __function_call__(self, messages: List[Dict], model: str,
                                tools: List[Dict], **kwargs):
        await asyncio.sleep(
            random.choice([0.4, 0.5, 0.6, 0.9, 1, 0.3, 0.2, 0.1]))
        output = await self.client.chat.completions.create(
            messages=messages,
            model=model,
            tools=tools,
            tool_choice=kwargs.get("tool_choice"))
        usage = output.usage.__dict__
        output_message = output.choices[0].message
        tool_calls = json.loads(
            output_message.tool_calls[0].function.arguments)
        return tool_calls, usage

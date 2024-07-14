from typing import List, Dict, Union
from transformers import AutoTokenizer


class MessageManagement:

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it",
                                                       token="")

    def __count_tokens__(self, content: str):
        return len(self.tokenizer.encode(content)) + 2

    def __pad_message__(self, content: str, num_tokens: int):
        return self.tokenizer.decode(
            self.tokenizer.encode(content)[:num_tokens])

    def __call__(self, messages: List[Dict], max_length: int = 3500):
        used_tokens = 0
        managed_messages = []
        system_prompt = list(
            filter(lambda message: message.get("role") == "system", messages))
        if system_prompt and len(system_prompt) > 0:
            used_tokens += self.__count_tokens__(
                system_prompt[0].get("content"))
        previous_role = None
        for ix, message in enumerate(messages[::-1]):
            content = message.get("content")
            current_role = message.get("role")
            content_tokens = self.__count_tokens__(content)
            if content_tokens + used_tokens >= max_length:
                content_tokens = max_length - used_tokens
                if content_tokens <= 0:
                    break
                content = self.__pad_message__(content, content_tokens)
            if current_role == previous_role:
                managed_messages[-1]["content"] += f"\n{content}"
            else:
                managed_messages += [{
                    "role": current_role,
                    "content": content
                }]
            used_tokens += content_tokens
            previous_role = current_role
        managed_messages = managed_messages[::-1]
        if system_prompt and len(system_prompt) > 0:
            managed_messages = system_prompt + managed_messages
        return managed_messages


if __name__ == "__main__":
    # ctx = MessageManagement("philschmid/gemma-tokenizer-chatml")
    ctx = MessageManagement("google/gemma-2b-it")
    messages = [
        {
            "role": "system",
            "content": "You are Gemma."
        },
        {
            "role": "user",
            "content": "Hello, how are you?"
        },
        {
            "role": "assistant",
            "content": "I'm doing great. How can I help you today?"
        },
    ]
    managedOutput = ctx(messages)
    print(managedOutput)

from abc import ABC, abstractmethod
from typing import List, Dict, Union


class BaseLLM(ABC):

    def __init__(self, **kwargs):
        self.client = None

    @abstractmethod
    async def __complete__(self, messages: List[Dict], model: str, **kwargs):
        pass

    @abstractmethod
    async def __stream__(self, messages: List[Dict], model: str, **kwargs):
        pass

    @abstractmethod
    async def __function_call__(self, messages: List[Dict], model: str,
                                **kwargs):
        pass

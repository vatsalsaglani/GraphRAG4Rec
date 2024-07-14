from typing import List, Dict


def calculateUsages(usages: List[Dict]):
    usages = {
        "completion_tokens": sum([u.get("completion_tokens") for u in usages]),
        "prompt_tokens": sum([u.get("prompt_tokens") for u in usages]),
        "total_tokens": sum([u.get("total_tokens") for u in usages])
    }
    return usages

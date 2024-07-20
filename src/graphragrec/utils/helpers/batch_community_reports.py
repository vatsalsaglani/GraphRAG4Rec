import json
import tiktoken
from typing import Dict

MAX_TOKEN_LENGTH = 5100
MODEL = "gpt-4o"
MAX_ALLOWED_MODEL_CTX = 100_000
MAX_COMMUNITIES_IN_BATCH = 8

print(f'LOADING ENCODING')
encoding = tiktoken.get_encoding("o200k_base")
print(f'LOADED ENCODING')


def batchCommunityReports(community_report: Dict[str, Dict]):
    community_batches = []
    community_ids = list(community_report.keys())
    community_ids = list(
        filter(lambda cid: len(community_report[cid].get("data")) > 0,
               community_ids))
    index = 0
    while index < len(community_ids):
        current_token_nums = 0
        community_batches.append({})
        while current_token_nums < MAX_TOKEN_LENGTH:
            if len(list(
                    community_batches[-1].keys())) >= MAX_COMMUNITIES_IN_BATCH:
                break
            community_id = community_ids[index]
            report = community_report[community_id].get("report")
            report_text = f'{report}'
            report_tokens = len(encoding.encode(report_text))
            if current_token_nums + report_tokens > MAX_TOKEN_LENGTH:
                if report_tokens < MAX_ALLOWED_MODEL_CTX:
                    community_batches.append({})
                    community_batches[-1][community_id] = report
                    index += 1
                    break
                else:
                    raise Exception(
                        f"A single community summary cannot be greater than {MAX_ALLOWED_MODEL_CTX} tokens!"
                    )
            else:
                community_batches[-1][community_id] = report
                current_token_nums += report_tokens
                index += 1
            if index >= len(community_ids):
                break
    return community_batches


if __name__ == "__main__":
    community_report = json.loads(
        open("./output/v9-gpt-4o-mini/community-reports.json").read())
    community_batches = batchCommunityReports(community_report)
    print(f"TOTAL BATCHES: {len(community_batches)}")
    for ix, batch in enumerate(community_batches):
        print(
            f"BATCH: {ix} has {len(list(batch.keys()))} community reports. | COMMUNITIES: [{', '.join(list(batch.keys()))}]"
        )
    with open("./output/v9-gpt-4o-mini/batched-community-reports.json",
              "w") as fp:
        json.dump(community_batches, fp, indent=4)

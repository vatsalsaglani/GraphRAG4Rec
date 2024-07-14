import os
import time
import json
import random
import logging
import asyncio
import tiktoken
import networkx as nx
from typing import Dict, List, Union
from tqdm.asyncio import tqdm
from tqdm.auto import trange
from llm.localllm import LocalLLM
from graphragrec.embed.community.report import communityReport
from graphragrec.embed.community.summary import combineCommunityReports
from graphragrec.utils.usage import calculateUsages

logging.basicConfig(filename='movie_community_embedding_1.log',
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

encoding = tiktoken.encoding_for_model("gpt-3.5-turbo-0125")


def fetchCommunityData(G: nx.Graph, communities: Dict):
    community2data = {}
    for key, value in communities.items():
        if not value in community2data:
            community2data[value] = []
        if key in G:
            node_data = {
                neighbor: dict(G[key][neighbor])
                for neighbor in G[key]
            }
            community2data[value].append({
                "entity": key,
                "relations-claims": node_data
            })
        else:
            print(f"{key} not in graph")
    return community2data


def divideCommunity(community_content: List):
    index = 0
    community_batch = []
    while index < len(community_content):
        community_batch.append([])
        token_length = 0
        # print("TOKEN LENGTH: ", token_length)
        while token_length < 10_000:
            token_length += len(encoding.encode(f'{community_content[index]}'))
            # print("TOKEN LENGTH: ", token_length)
            if token_length < 10_000:
                community_batch[-1].append(community_content[index])
                index += 1
            else:
                break
            if index >= len(community_content):
                break
    return community_batch


def batchCommunities(community_data: Dict):
    communitybatches = {}
    for cid, items in community_data.items():
        community_batches = divideCommunity(items)
        communitybatches[cid] = community_batches
    return communitybatches


async def summarizeCommunity(llm: LocalLLM, model: str, community_id: int,
                             community_batches: List[List[Dict]]):
    if len(community_batches) == 1:
        cr, usage = await communityReport(llm, model, community_batches)
        if isinstance(cr, Dict):
            cr["community_id"] = community_id
        return cr, usage
    pbar = tqdm(total=len(community_batches),
                desc=f"Community Batch - {community_id}",
                colour="blue")
    community_reports = []
    usages = []
    batch_size = 2
    for ix in range(0, len(community_batches), batch_size):
        crs = await asyncio.gather(*[
            communityReport(llm, model, community_batches[c])
            for c in range(ix, ix + batch_size, 1)
            if c < len(community_batches)
        ])
        # logging.info(f"CRS: {crs}")
        u = calculateUsages([c[-1] for c in crs])
        usages.append(u)
        community_reports.extend([c[0] for c in crs])
        logging.info("COMMUNITY COOLDOWN>>>>>>")
        time.sleep(20)
        logging.info("COMMUNITY COOLDOWN DONE>>>>>>")
        pbar.update(batch_size)
    cr, usage = await combineCommunityReports(llm, model, community_reports)
    usages += [usage]
    usages = calculateUsages(usages)
    if isinstance(cr, Dict):
        cr["community_id"] = community_id
    return cr, usages


async def summarizeCommunities(llm: LocalLLM, model: str, communities: Dict):
    communities_batched = batchCommunities(communities)
    pbar = tqdm(total=len(communities_batched),
                desc="Summarizing Communities",
                colour="blue")
    usages = []
    comprehensive_communities_reports = {}
    community_ids = list(communities.keys())

    batch_size = 5
    flatten = lambda lst: [item for sublist in lst for item in sublist]

    async def execute_community_summary(community_id: int,
                                        community_batches: List[List[Dict]]):
        try:
            time.sleep(random.choice([0.2, 0.4, 0.5, 0.8, 0.9, 1.1]))
            cr, u = await summarizeCommunity(llm, model, community_id,
                                             community_batches)
            comprehensive_communities_reports[community_id] = {
                "data": flatten(community_batches),
                "report": cr
            }
            usages.append(u)
            pbar.update(1)
        except Exception as err:
            logging.exception(
                f"EXCEPTION: {str(err)}\nCOMMUNITY: {community_id}")
            pbar.update(1)

    async for ix in trange(0,
                           len(community_ids),
                           batch_size,
                           desc="Batch Summarization",
                           leave=False):
        try:
            await asyncio.gather(*[
                execute_community_summary(
                    community_ids[i], communities_batched[community_ids[i]])
                for i in range(ix, ix + batch_size, 1)
                if i < len(community_ids)
            ])
        except Exception as err:
            logging.exception(f"Exception: {str(err)}")
            pbar.update(1)
        logging.info("COOLDOWN>>>>")
        time.sleep(30)
        logging.info("COOLDOWN DONE>>>>")
    total_usage = calculateUsages(usages)
    return comprehensive_communities_reports, total_usage


if __name__ == "__main__":
    import pickle
    import json
    from configs import OPENAI_API_KEY
    with open("./output/v7-all/graph.gpickle", "rb") as fp:
        G = pickle.load(fp)
    with open("./output/v7-all/communities.json", "r") as fp:
        communities = json.load(fp)
    community2data = fetchCommunityData(G, communities)
    with open("./output/v7-all/community-data.json", "w") as fp:
        json.dump(community2data, fp, indent=4)
    # commnuity_batches = batchCommunities(community2data)
    # print(json.dumps(commnuity_batches, indent=4))
    llm = LocalLLM(api_key=OPENAI_API_KEY)
    community_reports, total_usage = asyncio.run(
        summarizeCommunities(llm, "gpt-3.5-turbo-0125", community2data))
    with open("./output/v7-all/community-reports.json", "w") as fp:
        json.dump(community_reports, fp, indent=4)

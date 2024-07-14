import os
import asyncio
import pickle
import traceback
import logging
import json
import networkx as nx
from typing import List, Dict
from tqdm.asyncio import tqdm
from tqdm.auto import trange

from graphragrec.extract.claim import extractClaims
from graphragrec.extract.entities import extractEntities
from graphragrec.extract.relation import extractRelations
from llm.localllm import LocalLLM
from graphragrec.utils.usage import calculateUsages

logging.basicConfig(filename='movie_graph_builder_2.log',
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def save_graph_data(graph_data: List[Dict], file_path: str):
    with open(file_path, "w") as fp:
        json.dump(graph_data, fp)


def save_graph(G: nx.Graph, file_path: str):
    with open(file_path, "wb") as fp:
        pickle.dump(G, fp)


async def create_graph_data(llm: LocalLLM, model: str, movies: List[Dict]):
    # G = nx.Graph()
    usages = []
    graph_data = []
    semaphore = asyncio.Semaphore(5)
    pbar = tqdm(total=len(movies), desc="Processing Movies", colour="blue")

    async def process_movie_with_semaphore(movie: Dict):
        async with semaphore:
            try:
                await process_movie(movie.get("title"), movie.get("llm_text"))
                pbar.update(1)
            except Exception as err:
                logging.exception(f"EXCEPTION: {str(err)}")
                pbar.update(1)

    async def process_movie(movie: str, movie_overview: str):
        entities, eu = await extractEntities(llm, model, movie_overview)
        relations, ru = await extractRelations(llm, model, entities)
        claims, cu = await extractClaims(llm, model, movie_overview, entities,
                                         relations)
        usages.extend([eu, ru, cu])
        graph_data.append({
            "movie": movie,
            "overview": movie_overview,
            "entities": entities,
            "relations": relations,
            "claims": claims
        })

    await asyncio.gather(
        *[process_movie_with_semaphore(movie) for movie in movies])

    return graph_data, usages


async def batch_create_graph_data(llm: LocalLLM, model: str,
                                  movies: List[Dict]):
    usages = []
    graph_data = []
    batch_size = 100
    cooldown_time = 20
    async for ix in trange(0,
                           len(movies),
                           batch_size,
                           colour="green",
                           desc="Batch Graph"):
        g, u = await create_graph_data(llm, model, movies[ix:ix + batch_size])
        graph_data += g
        usages += u
        logging.info(f"Cooldown...")
        await asyncio.sleep(cooldown_time)
        logging.info(f"Cooldown Complete")
    return graph_data, usages


async def build_multi_movie_graph(llm: LocalLLM, model: str,
                                  movies: List[Dict]):
    graph_data, usages = await batch_create_graph_data(llm, model, movies)
    G = nx.Graph()

    for _, item in enumerate(tqdm(graph_data, desc="Creating Graph")):
        try:
            entities = item.get("entities")
            relations = item.get("relations")
            claims = item.get("claims")
            movie = item.get("movie")
            movie_overview = item.get("overview")

            G.add_node(movie, type="Movie", overview=movie_overview)

            for entity in entities:
                if not G.has_node(entity["name"]):
                    G.add_node(entity["name"], **entity)
                G.add_edge(movie, entity["name"], type="HAS_ENTITY")

            for relation in relations:
                if G.has_edge(relation["source"], relation["target"]):
                    G[relation["source"]][relation["target"]].update(relation)
                else:
                    G.add_edge(relation["source"], relation["target"],
                               **relation)

            for claim in claims["entity"]:
                if claim["subject"] in G.nodes:
                    if "claims" not in G.nodes[claim["subject"]]:
                        G.nodes[claim['subject']]['claims'] = []
                    G.nodes[claim['subject']]['claims'].append({
                        'predicate':
                        claim['predicate'],
                        'object':
                        claim['object'],
                        'confidence':
                        claim['confidence']
                    })

            for claim in claims["relation"]:
                if G.has_edge(claim['subject'], claim['object']):
                    if 'claims' not in G[claim['subject']][claim['object']]:
                        G[claim['subject']][claim['object']]['claims'] = []
                    G[claim['subject']][claim['object']]['claims'].append({
                        'predicate':
                        claim['predicate'],
                        'value':
                        claim.get('value', ''),
                        'confidence':
                        claim['confidence']
                    })
        except Exception as err:
            logging.exception(
                f"EXCEPTION: {str(err)}\n{json.dumps(item, indent=4)}\nTRACEBACK: {traceback.format_exc()}"
            )

    return G, graph_data, usages


async def build_and_save_multi_movie_graph(llm: LocalLLM, model: str,
                                           movies: List[Dict], file_path: str):
    if os.path.exists(os.path.join(file_path, "graph.json")):
        graph_path = os.path.join(file_path, "graph.gpickle")
        graph_data_path = os.path.join(file_path, "graph.json")
        with open(graph_path, "rb") as fp:
            G = pickle.load(fp)
        with open(graph_data_path, "r") as fp:
            graph_data = json.load(fp)
        return G, graph_data
    G, graph_data, usages = await build_multi_movie_graph(llm, model, movies)
    graph_path = os.path.join(file_path, "graph.gpickle")
    save_graph(G, graph_path)
    graph_data_path = os.path.join(file_path, "graph.json")
    save_graph_data(graph_data, graph_data_path)
    usages = calculateUsages(usages)
    logging.info(f'TOTAL TOKEN USAGE: {usages}')
    return G, graph_data


if __name__ == "__main__":
    from configs import OPENAI_API_KEY
    from graphragrec.utils.visualize_graph import visualize_graph
    from graphragrec.utils.detect_communities import detect_communities, visualize_community_graph, save_communities
    movies = json.loads(open("./imdb/data/imdb_top_100.json").read())
    use_movies = movies
    llm = LocalLLM(api_key=OPENAI_API_KEY)
    file_path = "./output/v7-all"
    if not os.path.exists(file_path):
        os.mkdir(file_path)
    G, graph_data = asyncio.run(
        build_and_save_multi_movie_graph(llm, "gpt-3.5-turbo-0125", use_movies,
                                         file_path))
    visualize_graph(G, file_path)
    communities = detect_communities(G)
    save_communities(communities, file_path)
    visualize_community_graph(G, communities, file_path)

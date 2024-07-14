import os
import json
# import json_repair as json
import networkx as nx
import leidenalg as la
import igraph as ig
from typing import List, Dict
from pyvis.network import Network
import matplotlib.colors as mcolors
import matplotlib.cm as cm


def detect_communities(G: nx.Graph):
    ig_graph = ig.Graph.from_networkx(G)
    partition = la.find_partition(ig_graph, la.ModularityVertexPartition)
    communities = {
        ig_graph.vs[node]['name']: membership
        for membership, community in enumerate(partition)
        for node in community
    }
    return communities


def save_communities(communities, file_path):
    file_path = os.path.join(file_path, "communities.json")
    with open(file_path, 'w') as fp:
        json.dump(communities, fp)


def visualize_community_graph(graph: nx.Graph, communities: Dict,
                              output_file: str):
    file_path = os.path.join(output_file, "community-graph.html")
    net = Network(notebook=True, height="800px", width="100%")
    num_communities = len(set(communities.values()))
    print(f'TOTAL COMMUNITIES: ', num_communities)
    color_map = cm.get_cmap("tab20", num_communities)

    for node, data in graph.nodes(data=True):
        community_id = communities.get(node, 0)
        color = mcolors.rgb2hex(color_map(community_id % num_communities))
        title = f"{node}<br>Type: {data.get('type', '')}<br>Overview: {data.get('overview', '')}<br>Community: {community_id}<br>"
        if 'claims' in data:
            claims = "<br>".join([
                f"{claim['predicate']} - {claim['object']} (Confidence: {claim['confidence']})"
                for claim in data['claims']
            ])
            title += f"Claims: <br>{claims}"
        net.add_node(node, title=title, color=color, **data)

    for source, target, data in graph.edges(data=True):
        title = f"Type: {data.get('type', '')}<br>"
        if 'claims' in data:
            claims = "<br>".join([
                f"{claim['predicate']} - {claim['value']} (Confidence: {claim['confidence']})"
                for claim in data['claims']
            ])
            title += f"Claims: <br>{claims}"
        if "source" in data:
            del data["source"]
        if "target" in data:
            del data["target"]
        net.add_edge(source, target, title=title, **data)

    net.show(file_path)

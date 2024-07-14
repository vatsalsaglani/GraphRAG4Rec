import os
import json
import networkx as nx
from pyvis.network import Network


def visualize_graph(graph: nx.Graph, file_path):
    file_path = os.path.join(file_path, "rag-graph.html")
    net = Network(notebook=True, height="800px", width="100%")

    for node, data in graph.nodes(data=True):
        title = f"{node}<br>Type: {data.get('type')}<br>Overview: {data.get('overview', '')}<br>"
        if 'claims' in data:
            claims = "<br>".join([
                f"{claim['predicate']} - {claim['object']} (Confidence: {claim['confidence']})"
                for claim in data["claims"]
            ])
            title += f"Claims: <br>{claims}"
        net.add_node(node, title=title, **data)

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

"""
Run prim and kruskal algorithms and compare time they takes for a different
number of nodes
"""
import random
import networkx as nx
import time
from tqdm import tqdm

from itertools import combinations, groupby

NUM_OF_ITERATIONS = 3
chance = 0.5


def gnp_random_connected_graph(num_of_nodes, completeness):
    """
    Generates a random undirected graph, similarly to an Erdős-Rényi
    graph, but enforcing that the resulting graph is connected
    """
    edges = combinations(range(num_of_nodes), 2)
    G = nx.Graph()
    G.add_nodes_from(range(num_of_nodes))
    for _, node_edges in groupby(edges, key=lambda x: x[0]):
        node_edges = list(node_edges)
        random_edge = random.choice(node_edges)
        G.add_edge(*random_edge)
        for e in node_edges:
            if random.random() < completeness:
                G.add_edge(*e)
    for (u, v, w) in G.edges(data=True):
        w['weight'] = random.randint(0, 10)
    return G


def prim_algorithm(graph, starting_node=0):
    """
    Find minimum spanning tree by prim algorithm
    :param graph: undirected graph, similarly to an Erdős-Rényi graph, but
    enforcing that the resulting graph is connected
    :param starting_node: number of node from which you want to start building
    graph
    :return: minimum spanning tree, list of edges with it's weight
    """

    def extracting_graph_prim(g):
        """
        Get values from graph
        :param g: graph
        :return: edges, nodes and ways to vertices
        """
        vertices_edges = []
        ways_to_vertices = []
        i = 0
        for _ in g.nodes():
            vertices_edges.append(set())
            ways_to_vertices.append({})
            i += 1
        for (u, v, w) in g.edges(data=True):
            vertices_edges[u].add(tuple([u, v, w["weight"]]))
            vertices_edges[v].add(tuple([u, v, w["weight"]]))
        return vertices_edges, i, ways_to_vertices

    vertices_edges, nodes, ways_to_vertices = extracting_graph_prim(graph)
    nuv_set = set(range(1, nodes))
    last_uv = starting_node
    used_edges = []
    while len(used_edges) < nodes - 1:
        for edge in vertices_edges[last_uv]:
            if edge[0] in nuv_set:
                if not ways_to_vertices[edge[0]] or \
                        ways_to_vertices[edge[0]][2] > edge[2]:
                    ways_to_vertices[edge[0]] = edge
            elif edge[1] in nuv_set:
                if not ways_to_vertices[edge[1]] or \
                        ways_to_vertices[edge[1]][2] > edge[2]:
                    ways_to_vertices[edge[1]] = edge
        min_edge = 0
        for node in nuv_set:
            if ways_to_vertices[node] and (not min_edge or min_edge[2] >
                                           ways_to_vertices[node][2]):
                min_edge = ways_to_vertices[node]

        last_uv = min_edge[0] if min_edge[0] in nuv_set else min_edge[1]
        nuv_set.remove(last_uv)
        min_edge = list(min_edge)
        min_edge[2] = {'weight': min_edge[2]}
        used_edges.append(min_edge)
    return used_edges


def kruskal_algorithm(graph):
    """
    Find minimum spanning tree by kruskal algorithm
    :param graph: undirected graph, similarly to an Erdős-Rényi graph, but
    enforcing that the resulting graph is connected
    :return: minimum spanning tree, list of edges with it's weight
    """
    vertices_count = len(graph.nodes)
    edges = list(graph.edges(data=True))
    edges = sorted(edges, key=lambda x: x[2]['weight'])
    vertices = []
    for iterator in range(0, vertices_count):
        vertices.append({iterator})
    itr = 0
    tree = []
    while len(tree) < vertices_count - 1:
        pos_0 = edges[itr][0]
        pos_1 = edges[itr][1]
        while isinstance(vertices[pos_0], int):
            pos_0 = vertices[pos_0]
        while isinstance(vertices[pos_1], int):
            pos_1 = vertices[pos_1]
        if vertices[pos_0] == vertices[pos_1]:
            itr += 1
            continue
        else:
            vertices[pos_0] = vertices[pos_0] | vertices[pos_1]
            vertices[pos_1] = pos_0
            tree.append(edges[itr])
            itr += 1
    return tree


def measuring_time(func):
    """
    Calculate time spent on function
    :param func: function
    """
    time_dict = {}
    for num_of_nodes in [5, 10, 20, 50, 100, 200, 500, 1000]:
        time_taken = 0
        for i in tqdm(range(NUM_OF_ITERATIONS)):
            # note that we should not measure time of graph creation
            G = gnp_random_connected_graph(num_of_nodes, chance)
            start = time.time()
            func(G)
            end = time.time()

            time_taken += end - start

        time_dict[num_of_nodes] = time_taken / NUM_OF_ITERATIONS
    print(time_dict)


if __name__ == "__main__":
    print("Time spent on prim algorithm:")
    measuring_time(prim_algorithm)
    print("Time spent on kruskal algorithm:")
    measuring_time(kruskal_algorithm)

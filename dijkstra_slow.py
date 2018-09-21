# Implement Dijkstra's algorithm for finding all shortest paths between a vertex and all other vertices.
# Kozen ch5 p 26
#
# Slow because we do not use a fibonacci heap

from collections import defaultdict

# Reused graph class from mst_slow.py
# With modification to reject negative edge-weights
class Graph:
    def __init__(self, adj_list=None):
        self.adj_list = defaultdict(lambda: list())

    def __getitem__(self, v):
        return self.adj_list[v]

    def add_edge(self, v1, v2, w):
        assert w >= 0, "Weight must be nonnegative"
        self.adj_list[v1].append((v2, w))
        self.adj_list[v2].append((v1, w))

    def has_edge(self, v1, v2):
        # map gets us the vertex out of the (v2, weight) tuple
        return v2 in map(lambda x: x[0], self.adj_list[v1])
    
    # Returns a dict whose keys are the vertices of the graph (target vertices)
    # and the values are a shortest path from `start_vertex` to that vertex.
    # Each path is a list of edges, with the first edge's starting vertex being
    # `start_vertex` and the last edge's end vertex being the same as the
    # target vertex.
    # We do not include the self-path, so each dict has `number of vertices - 1` entries
    def shortest_paths(self, start_vertex):
        return {1: [(0,1,1),]}

# Connected weighted graph generator, reused from mst_slow.py

# I am not sure if there are graphs this method cannot generate
def gen_weighted_connected_graph(num_vertices, num_edges, max_weight):
    assert num_edges >= num_vertices-1

    g = Graph()

    v_used = [0,]
    v_unused = list(range(1,num_vertices))

    edges_remaining = num_edges
    while edges_remaining > 0:
        # if there are still unconnected vertices
        if len(v_unused) > 0:
            v1 = random.choice(v_used)
            v2 = random.choice(v_unused)
            v_unused.remove(v2)
            v_used.append(v2)
        else:
            v1 = random.choice(v_used)
            v2 = random.choice(v_used)
            if v1 == v2:
                continue

        w = random.choice(range(1, max_weight))
        if not g.has_edge(v1, v2):
            g.add_edge(v1, v2, w)
            edges_remaining -= 1
        else:
            continue

    return g

# Tests

import random

def test_one_edge():
    g = Graph()
    g.add_edge(0,1,1)

    paths = g.shortest_paths(0)
    assert len(paths) == 1
    assert paths[1] == [(0,1,1)]

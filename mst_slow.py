# Given a connected, weighted, undirected graph G, compute a minimum spanning tree of G. That is, compute a tree T such that each vertex of G is in thetree T, and the total weight of the tree is minimized.
# Kozen 2.1 p 11, Kruskal's greedy algorithm

# Currently the tests only check that it is a spanning tree, not a minimal one. I am not sure how to check that it is minimal.
# This is a "slow" implementation because the merge operation is slow - we do not use a union-find data structure.

from collections import defaultdict

class Graph:
    def __init__(self, adj_list=None):
        self.adj_list = defaultdict(lambda: list())

    def __getitem__(self, v):
        return self.adj_list[v]

    def add_edge(self, v1, v2, w):
        self.adj_list[v1].append((v2, w))
        self.adj_list[v2].append((v1, w))

    def has_edge(self, v1, v2):
        # map gets us the vertex out of the (v2, weight) tuple
        return v2 in map(lambda x: x[0], self.adj_list[v1])

    # Returns a set of edges such that for each tuple (v1, v2, w), v1 < v2
    def edges(self):
        edges = set()
        for v1, neighbors in self.adj_list.items():
            for v2, w in neighbors:
                if v1 < v2:
                    edges.add((v1,v2,w))
        return edges

    # determines if graph is connected via simple BFS
    def is_connected(self):
        visited = set()
        queue = list()
        queue.append(0)
        while queue:
            current = queue.pop()
            visited.add(current)
            for n,_ in self.adj_list[current]:
                if n not in visited:
                    queue.append(n)

        # if we visited all the vertices after a BFS then it's connected
        return len(visited) == len(self.adj_list)

    # Returns a new graph which is a minimum spanning tree of the current graph
    def mst_slow(self):
        g = Graph()
        edges = self.edges()

        # sort edges by weight
        edges = sorted(edges, key = lambda x: x[2])

        components = dict()
        new_component = 0

        print(components)
        for edge in edges:
            print(components)
            v1,v2 = edge[0], edge[1]
            # both vertices are in the same component: skip it
            if components.get(v1) == components.get(v2) != None:
                continue
            # neither vertex is part of a component: make new component
            elif v1 not in components and v2 not in components:
                components[v1] = new_component
                components[v2] = new_component
                new_component+=1
            # only one vertex in the edge is part of an existing component: add
            # other to existing component
            elif v1 in components and v2 not in components:
                components[v2] = components[v1]
            elif v2 in components and v1 not in components:
                components[v1] = components[v2]
            # both vertices are in different components: merge them
            # this section is why this is mst_slow
            else:
                c1 = components[v1]
                c2 = components[v2]
                # we can make this faster using a union-find data structure
                # 
                # I don't know if it ends up being the same, but I think if we
                # add another layer of indirection by making the component
                # numbers into objects, then we don't have to iterater through
                # and update everything: we just update once and all the others
                # are updated automatically 
                for v,c in components.items():
                    if c == c2:
                        components[v] = c1
            g.add_edge(edge[0], edge[1], edge[2])
        return g


# Connected weighted graph generator

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

        w = random.choice(range(0, max_weight))
        if not g.has_edge(v1, v2):
            g.add_edge(v1, v2, w)
            edges_remaining -= 1
        else:
            continue

    return g


# Tests

import random

## MST tests

def test_two_vertices():
    g = Graph()
    g.add_edge(0, 1, 1)

    mst = g.mst_slow()

    assert len(mst.adj_list) == len(g.adj_list)
    assert_is_tree(g.mst_slow())

def test_random_graphs_mst_slow():
    for _ in range(0,100):
        n = 50
        num_edges = random.choice(range(n - 1, n*(n-1)//2))
        max_weight = 100
        g = gen_weighted_connected_graph(n, num_edges, max_weight)
        mst = g.mst_slow()

        assert len(mst.adj_list) == len(g.adj_list)
        assert_is_tree(mst)

## assertion functions
def assert_is_tree(graph):
    # a tree with n vertices has n-1 edges, and conversely if a graph is
    # connected and has n-1 edges it must be a tree
    assert len(graph.edges()) == (len(graph.adj_list) - 1)
    assert_is_connected(graph)

def assert_is_connected(graph):
    assert graph.is_connected()

## graph utility function tests 
def test_is_connected():
    g = Graph()
    g.add_edge(0,1,1)
    g.add_edge(2,3,1)
    assert not g.is_connected()

    g = Graph()
    g.add_edge(0,1,1)
    g.add_edge(1,2,1)
    assert g.is_connected()

def test_random_graphs_edges():
    for _ in range(0,100):
        n = 50
        num_edges = random.choice(range(n - 1, n*(n-1)//2))
        max_weight = 100

        g = gen_weighted_connected_graph(n, num_edges, max_weight)

        assert len(g.edges()) == num_edges

def test_random_graphs_edges_small():
    for _ in range(0,100):
        n = 5
        num_edges = 7
        max_weight = 100

        g = gen_weighted_connected_graph(n, num_edges, max_weight)

        assert len(g.edges()) == num_edges

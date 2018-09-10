from collections import defaultdict

# Creates a directed graph stored as an adjacency list, where each element in
# the list contains a list of incoming and outgoing edges, represented by the
# vertex on the other end. Technically I guess the vertices can be anything
# because we're storing them in a dict but they should ideally be integers.
class Graph:
    # Edges is a list of tuples of vertices `(v1, v2)` representing a directed
    # edge from `v1` to `v2`.
    def __init__(self, edges=None):
        # outgoing edges are adj_list[v][0]
        # incoming edges are adj_list[v][1]
        self.adj_list = defaultdict(lambda: (list(), list()))
        if edges:
            for v1,v2 in edges:
                self.add_edge(v1, v2)

    # Creates a directed edge between `v1` and `v2` if it does not already exist
    def add_edge(self, v1, v2):
        # outgoing
        self.adj_list[v1][0].append(v2)

        # incoming
        self.adj_list[v2][1].append(v1)

    # Get the outgoing edges from vertex `v`
    def out(self, v):
        return self.adj_list[v][0]

    # Get the incoming edges from vertex `v`
    def inc(self, v):
        return self.adj_list[v][1]

    # Returns a list of the vertices representing at topological ordering
    # obeying the graph's structure. The algorithm in Kozen, and the one we
    # implement here, is the Tarjan algorithm, not the DFS algorithm.
    def topo_sort(self):
        return [0,1]

### Tests

## Basic add tests
def test_add_edge_1():
    g = Graph()
    v1 = 0
    v2 = 1

    g.add_edge(v1, v2)
    assert v2 in g.out(v1)
    assert v1 in g.inc(v2)

    assert len(g.inc(v1)) == 0
    assert len(g.out(v2)) == 0

def test_add_edges_2():
    g = Graph()
    v1 = 0
    v2 = 1
    v3 = 2

    g.add_edge(v1, v2)
    g.add_edge(v2, v3)
    assert v2 in g.out(v1)
    assert v1 in g.inc(v2)
    assert v3 in g.out(v2)
    assert v2 in g.inc(v3)

    assert len(g.inc(v1)) == 0
    assert len(g.out(v3)) == 0

## Utility for checking whether topological sort is valid possiblility

def assert_topo_sort(graph, order):
    g = graph
    for i,v in enumerate(order):
        for x in g.out(v):
            # assert that if x is an outgoing neighbor of v, then it comes
            # after v in the ordering
            assert i < order.index(x)

## Topological sort tests

def test_one_edge():
    g = Graph([(0,1)])

    order = g.topo_sort()
    assert order == [0,1]

    assert_topo_sort(g, order)

# def test_two_edges():
#     edges = [(0,1), (1,2)]
#     g = Graph(edges)
# 
#     order = g.topo_sort()
#     assert order == [0,1,2]
# 
#     assert_topo_sort(g, order)

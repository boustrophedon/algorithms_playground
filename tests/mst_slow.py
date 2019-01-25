import random

from algo.mst_slow import Graph

# Connected weighted graph generator

# I am not sure if there are graphs this method cannot generate
def gen_weighted_connected_graph(num_vertices, num_edges, max_weight):
    assert num_edges >= num_vertices - 1

    g = Graph()

    v_used = [0]
    v_unused = list(range(1, num_vertices))

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


## MST tests


def test_two_vertices():
    g = Graph()
    g.add_edge(0, 1, 1)

    mst = g.mst_slow()

    assert len(mst.adj_list) == len(g.adj_list)
    assert_is_tree(g.mst_slow())


def test_triangle():
    g = Graph()
    g.add_edge(0, 1, 1)
    g.add_edge(0, 2, 1)
    g.add_edge(1, 2, 2)

    mst = g.mst_slow()

    assert len(mst.adj_list) == len(g.adj_list)
    assert_is_tree(g.mst_slow())


def test_random_graphs_mst_slow():
    for _ in range(0, 100):
        n = 50
        num_edges = random.choice(range(n - 1, n * (n - 1) // 2))
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
    g.add_edge(0, 1, 1)
    g.add_edge(2, 3, 1)
    assert not g.is_connected()

    g = Graph()
    g.add_edge(0, 1, 1)
    g.add_edge(1, 2, 1)
    assert g.is_connected()


def test_random_graphs_edges():
    for _ in range(0, 100):
        n = 50
        num_edges = random.choice(range(n - 1, n * (n - 1) // 2))
        max_weight = 100

        g = gen_weighted_connected_graph(n, num_edges, max_weight)

        assert len(g.edges()) == num_edges


def test_random_graphs_edges_small():
    for _ in range(0, 100):
        n = 5
        num_edges = 7
        max_weight = 100

        g = gen_weighted_connected_graph(n, num_edges, max_weight)

        assert len(g.edges()) == num_edges

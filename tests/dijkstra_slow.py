import random

from algo.dijkstra_slow import Graph

# Connected weighted graph generator, reused from mst_slow.py

# I am not sure if there are graphs this method cannot generate
# Update: I'm pretty sure you can prove that you can generate every connected
# graph with this method by induction but I am now equally sure that
# it doesn't generate them with equal probability.
#
# I think there are methods by which you first generate a spanning tree with
# uniform probability, and then add more edges. This method *does* generates a
# spanning tree first, but eg the first vertex we pick has a higher probability
# of having a higher degree than the last vertex picked.
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

        w = random.choice(range(1, max_weight))
        if not g.has_edge(v1, v2):
            g.add_edge(v1, v2, w)
            edges_remaining -= 1
        else:
            continue

    return g


def test_one_edge():
    g = Graph()
    g.add_edge(0, 1, 1)

    paths = g.shortest_paths(0)
    assert len(paths) == 1
    assert paths[1] == [(0, 1, 1)]


def test_two_edges_line():
    g = Graph()
    g.add_edge(0, 1, 1)
    g.add_edge(1, 2, 1)

    paths = g.shortest_paths(0)
    assert len(paths) == 2
    assert paths[1] == [(0, 1, 1)]
    assert paths[2] == [(0, 1, 1), (1, 2, 1)]


def test_two_edges_spoke():
    g = Graph()
    g.add_edge(0, 1, 1)
    g.add_edge(0, 2, 1)

    paths = g.shortest_paths(0)
    assert len(paths) == 2
    assert paths[1] == [(0, 1, 1)]
    assert paths[2] == [(0, 2, 1)]


def test_diamond_equal_weights():
    g = Graph()
    g.add_edge(0, 1, 1)
    g.add_edge(0, 2, 1)
    g.add_edge(1, 3, 1)
    g.add_edge(2, 3, 1)

    # this is implementation dependent but we could probably make it not so by
    # taking the lowest-numbered (or total-lowest-numbered? i haven't thought
    # this through) path
    paths = g.shortest_paths(0)
    assert len(paths) == 3
    assert paths[1] == [(0, 1, 1)]
    assert paths[2] == [(0, 2, 1)]
    assert paths[3] == [(0, 1, 1), (1, 3, 1)]


def test_diamond_unique_weights():
    g = Graph()
    g.add_edge(0, 1, 1)
    g.add_edge(0, 2, 1)
    g.add_edge(1, 3, 2)  # path that was selected in previous test is now higher-weight
    g.add_edge(2, 3, 1)

    paths = g.shortest_paths(0)
    assert len(paths) == 3
    assert paths[1] == [(0, 1, 1)]
    assert paths[2] == [(0, 2, 1)]
    assert paths[3] == [(0, 2, 1), (2, 3, 1)]


def test_triangle_shorter_total():
    g = Graph()
    g.add_edge(0, 1, 1)
    g.add_edge(1, 2, 1)

    g.add_edge(0, 2, 3)

    paths = g.shortest_paths(0)
    assert len(paths) == 2
    assert paths[1] == [(0, 1, 1)]
    assert paths[2] == [(0, 1, 1), (1, 2, 1)]


# TODO: determine how to property test/validate that paths are shortest
# easy sanity checks: path length is bounded by number of vertices (i.e. no repeated vertices)

# Implement Dijkstra's algorithm for finding all shortest paths between a vertex and all other vertices.
# Kozen ch5 p 26
#
# Slow because we do not use a fibonacci heap
# I decided to comment the steps of the algorithm somewhat better than previous algorithms

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

        # current candidates for next shortest-path vertex
        # this is the data structure that is the bottleneck for this algorithm.
        # to make it faster, we can use a fibonacci heap.
        queue = set()
        queue.add(start_vertex)

        # maps a vertex to the previous vertex along the shortest path to the start_vertex
        # we use this to compute the final shortest path at the end
        predecessors = dict()

        # maps a vertex to the distance along the shortest path from the start vertex
        distances = dict()
        distances[start_vertex] = 0

        # visited is not strictly necessary:
        # anything in put in visited will also not pass the "dist_to_neighbor < distances[n]" check.
        # of course, that means we use an extra O(|V|) memory, and the cost of
        # accessing that memory might be more than the check so in reality
        # you'd want to benchmark and be aware of the size of your data
        visited = set()

        while queue:
            # select minimum-length edge from margin
            # pop-min operation on margin data structure
            current = min(queue, key=lambda x: distances[x])
            queue.remove(current)

            # when we pop a vertex from the queue, we are guaranteed to know the
            # shortest path to that vertex. we mark it so that we don't need to
            # check it again.
            visited.add(current)

            for n, w in self.adj_list[current]:
                if n in visited:
                    continue

                dist_to_neighbor = distances[current] + w
                # if we haven't seen this vertex yet, or:
                # if the length of a path going through current vertex to the neighbor
                # is less than the length of the best currently-known path to the neighbor,
                # then replace the old path with the path going through the current vertex
                if n not in distances or dist_to_neighbor < distances[n]:
                    distances[n] = dist_to_neighbor
                    predecessors[n] = (current, w)
                    queue.add(n)

        # compute the output paths by following the predecessors backwards
        output = dict()
        for v in self.adj_list.keys():
            if v == start_vertex:
                continue

            current = v
            path = list()

            while current != start_vertex:
                prev = predecessors[current]
                path.append((prev[0], current, prev[1]))
                current = prev[0]

            output[v] = list(reversed(path))

        return output


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


# Tests

import random


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

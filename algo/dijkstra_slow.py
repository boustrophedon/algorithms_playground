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

from collections import defaultdict


# Creates a directed graph stored as an adjacency list, where each element in
# the list contains a list of incoming and outgoing edges, represented by the
# vertex on the other end. Technically I guess the vertices can be anything
# because we're storing them in a dict but they should ideally be integers.
class Graph:
    """ A directed graph with adjacency list/dict storage. Each element in its
    adjacency list contains a tuple of two lists, the first the outgoing edges
    and the second the incoming edges. The edges are represented by the
    index/element on the other end of the list.
    """

    def __init__(self, edges=None):
        """ Edges is a list of tuples of vertices `(v1, v2)` representing a
        directed edge from `v1` to `v2`.
        """
        # outgoing edges are adj_list[v][0]
        # incoming edges are adj_list[v][1]
        self.adj_list = defaultdict(lambda: (list(), list()))
        if edges:
            for v1, v2 in edges:
                self.add_edge(v1, v2)

    def add_edge(self, v1, v2):
        """ Creates a directed edge between `v1` and `v2` if it does not
        already exist. """
        # outgoing
        self.adj_list[v1][0].append(v2)

        # incoming
        self.adj_list[v2][1].append(v1)

    def out(self, v):
        """ Get the outgoing edges from vertex `v` """
        return self.adj_list[v][0]

    def inc(self, v):
        """ Get the incoming edges from vertex `v` """
        return self.adj_list[v][1]

    def topo_sort(self):
        """ Returns a list of the vertices representing at topological ordering
        obeying the graph's structure. The algorithm in Kozen, and the one we
        implement here, is the Kahn algorithm, not the DFS algorithm.

        This version is modified from the one in the book to not modify the graph
        in-place, at the expense of using O(n) more memory. """

        queue = list()
        order = list()

        inc_remaining = defaultdict(lambda: 0)

        for v, (out, inc) in self.adj_list.items():
            if len(inc) == 0:
                queue.append(v)
            else:
                inc_remaining[v] = len(inc)

        while queue:
            current = queue.pop()
            order.append(current)
            for v in self.out(current):
                inc_remaining[v] -= 1
                if inc_remaining[v] == 0:
                    queue.append(v)

        return order

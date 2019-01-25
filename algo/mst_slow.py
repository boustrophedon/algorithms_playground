# Given a connected, weighted, undirected graph G, compute a minimum spanning tree of G. That is, compute a tree T such that each vertex of G is in thetree T, and the total weight of the tree is minimized.
# Kozen 2.1 p 11, Kruskal's greedy algorithm

# Currently the tests only check that it is a spanning tree, not a minimal one. I am not sure how to check that it is minimal.
# This is a "slow" implementation because the merge operation is slow - we do not use a union-find data structure.

# Apparently there are linear time algorithms to do this, but then I'd have to write even more code to test the implementation of the verification algorithms.
# https://www.cs.princeton.edu/courses/archive/fall03/cs528/handouts/A%20Simpler%20Minimum%20Spanning.pdf

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
                    edges.add((v1, v2, w))
        return edges

    # determines if graph is connected via simple DFS
    def is_connected(self):
        visited = set()
        stack = list()
        stack.append(0)
        while stack:
            current = stack.pop()
            visited.add(current)
            for n, _ in self.adj_list[current]:
                if n not in visited:
                    stack.append(n)

        # if we visited all the vertices after a BFS then it's connected
        return len(visited) == len(self.adj_list)

    # Returns a new graph which is a minimum spanning tree of the current graph
    def mst_slow(self):
        g = Graph()
        edges = self.edges()

        # sort edges by weight
        edges = sorted(edges, key=lambda x: x[2])

        components = dict()
        new_component = 0

        for edge in edges:
            # we don't need the weight inside this loop
            v1, v2, _ = edge

            # both vertices are in the same component: skip it
            if components.get(v1) == components.get(v2) is not None:
                continue
            # neither vertex is part of a component: make new component
            elif v1 not in components and v2 not in components:
                components[v1] = new_component
                components[v2] = new_component
                new_component += 1
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
                for v, c in components.items():
                    if c == c2:
                        components[v] = c1
            g.add_edge(edge[0], edge[1], edge[2])
        return g

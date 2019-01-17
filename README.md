[![Build Status](https://travis-ci.org/boustrophedon/algorithms_playground.svg?branch=master)](https://travis-ci.org/boustrophedon/algorithms_playground)

Just practice implementing some algorithms for interview prep, mostly from Kozen's Design and Analysis of Algorithms. I started using the new typing system halfway through and would like to get the code passing `mypy --strict` at some point.

# Algorithms

[\*] indicates there are benchmarks for the code as well.

## Graph algorithms
- [x] Dijkstra's
- [ ] Dijkstra's with binomial/fibonacci heap
- [x] Kruskal's minimum spanning tree algorithm
- [ ] Kruskal's minimum spanning tree with a union-find datastructure
- [x] Kahn topological sort/ordering for directed graph
- [ ] Max-flow/min-cut
 	- [ ] Ford-Fulkerson
 	- [ ] Edmonds-Karp
- [ ] Matching algorithms?

## Hashmaps (not in Kozen)
- [ ] Closed addressing, separate chaining with lists
- [ ] Closed addressing, separate chaining with balanced binary trees (probably won't implement)
- [ ] Open addressing, linear probing
- [ ] Open addressing, quadratic probing
- [ ] Open addressing, Cuckoo hashing
- [ ] Open addressing, Hopscotch hashing
- [ ] Open addressing, Robin Hood hashing
- [ ] Maybe faster hash functions (many blog posts about this)

## Collections
- [x] Linked list just for fun
- [x] Binomial heaps
- [ ] Fibonacci heaps
- [ ] Pairing heap
- [ ] Treap
- [ ] Union-find
- [ ] Splay tree

## Misc
- [x] Strassen multiplication
- [x] Mergesort (non-recursive) [\*]
- [ ] Burrows-Wheeler transformation

# CI

Travis builds are gated on `black`, `flake8`, and `mypy`.

# Running the code

To run the tests for a given python file, have pipenv installed, install dependencies with `pipenv install --dev` and run `pipenv run pytest <python_file>`.

To run all tests you just run `pytest`.

To run individual benchmarks, you should be able to run `python -m algo.bench.bench_foo` inside the pipenv. I'm also experimenting with trying to test that a function matches a given Big-O model (e.g. mergesort is O(nlogn)) with some statistical significance tests, but it's a WIP because my stats are not that great. You can try running the tests with `pytest --bench`, but it takes a long time. See conftest.py for how the flag works, and algo/bench/merge.py for an example with mergesort.

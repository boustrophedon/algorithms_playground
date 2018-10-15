[![Build Status](https://travis-ci.org/boustrophedon/algorithms_practice.svg?branch=master)](https://travis-ci.org/boustrophedon/algorithms_practice)

Just practice implementing some algorithms for interview prep, mostly from Kozen's Design and Analysis of Algorithms.

To run the tests for a given python file, have pipenv installed, install dependencies with `pipenv install --dev` and run `pipenv run pytest <python_file>`.

To run all tests you just run `pytest`.

To run individual benchmarks, you should be able to run `python -m algo.bench.bench_foo` inside the pipenv. I'm also experimenting with trying to test that a function matches a given Big-O model (e.g. mergesort is O(nlogn)) with some statistical significance tests, but it's a WIP because my stats are not that great. You can try running the tests with `pytest --bench`, but it takes a long time. See conftest.py for how the flag works, and algo/bench/merge.py for an example with mergesort.

[![Build Status](https://travis-ci.org/boustrophedon/algorithms_practice.svg?branch=master)](https://travis-ci.org/boustrophedon/algorithms_practice)

Just practice implementing some algorithms from Kozen's Design and Analysis of Algorithms

Probably going to use Rust and Python.

To run the tests for a given python file, have pipenv installed, install dependencies with `pipenv install --dev` and run `pipenv run pytest <python_file>`.

To run all tests you can run `pytest algo`

To run individual benchmarks, you should be able to run `python -m algo.bench.bench_foo` but it's currently broken due to the way perf interact    s with packages or something.

import matplotlib.pyplot as plt

import csv
import timeit

from algo.mergesort import mergesort
from .models import Onlogn


def run_merge_benches_reversed(start_size, end_size, step_size, trials_per_array):
    test_arrays = [
        (n, list(reversed(range(0, n)))) for n in range(start_size, end_size, step_size)
    ]

    results = list()
    for n, v in test_arrays:
        result = timeit.timeit(lambda: mergesort(v), number=trials_per_array)
        results.append((n, result))

    return results


def test_bench_mergesort():
    results = run_merge_benches_reversed(100, 10000, 50, 100)

    x = [t[0] for t in results]
    y = [t[1] for t in results]

    model = Onlogn()
    model.fit(x, y)

    model.assert_good_params()
    assert model.score > 0.99


def show_mergesort_reversed_graph():
    print("Running benches")
    results = run_merge_benches_reversed(100, 10000, 50, 100)
    print("Saving data to /tmp/mergesort_reversed.csv")
    with open("/tmp/mergesort_reversed.csv", "w") as f:
        f.write("array size,execution time in seconds\n")

        writer = csv.writer(f)
        writer.writerows(results)

    print("Benches complete")
    x = [t[0] for t in results]
    y = [t[1] for t in results]

    plt.figure()
    plt.title("Mergesort on a reversed (n...0) array of integers")
    plt.xlabel("array size")
    plt.ylabel("runtime in seconds")
    plt.plot(x, y, "o-")

    print("Saving figure to /tmp/mergesort_reversed.png")
    plt.savefig("/tmp/mergesort_reversed.png")

    plt.show()


if __name__ == "__main__":
    show_mergesort_reversed_graph()

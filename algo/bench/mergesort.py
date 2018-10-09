import timeit
from algo.mergesort import mergesort
from algo.bench.models import Onlogn

import matplotlib.pyplot as plt

def run_merge_benches():
    test_arrays = [(n, list(reversed(range(0,n)))) for n in range(100, 3000, 50)]

    results = list()
    for n,v in test_arrays:
        result = timeit.timeit(lambda: mergesort(v), number=100)
        results.append((n, "mergesort_{}".format(n), result))

    return results

def test_bench_mergesort():
    results = run_merge_benches()

    x = [t[0] for t in results]
    y = [t[2] for t in results]

    # Uncomment to plot the data
    # plt.figure()
    # plt.xlabel("array size")
    # plt.ylabel("runtime in seconds")
    # plt.plot(x,y, "o-")
    # plt.show()

    model = Onlogn()
    model.fit(x,y)

    model.assert_good_params()
    assert model.score > 0.99

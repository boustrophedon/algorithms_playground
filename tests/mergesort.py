from hypothesis import given
import hypothesis.strategies as st

from algo.mergesort import mergesort, merge


def test_msort_empty():
    v = []
    result = mergesort(v)

    assert result == []


def test_msort_one():
    v = [1]
    result = mergesort(v)

    assert result == [1]


def test_msort_simple():
    v = [1, 2]
    result = mergesort(v)

    assert result == [1, 2]

    v = [2, 1]
    result = mergesort(v)

    assert result == [1, 2]


def test_msort_long():
    v = list(reversed(range(0, 10000)))
    result = mergesort(v)

    assert result == list(range(0, 10000))


def test_msort_small_cases():
    v = [1, 2, 3, 4]
    result = mergesort(v)

    assert result == [1, 2, 3, 4]

    v = [5, 4, 3, 2, 1]
    result = mergesort(v)

    assert result == [1, 2, 3, 4, 5]

    v = [1, 3, 2, 5, 4]
    result = mergesort(v)

    assert result == [1, 2, 3, 4, 5]

    v = [1, 2, 4, 5, 6, 2]
    result = mergesort(v)

    assert result == [1, 2, 2, 4, 5, 6]


@given(st.lists(st.integers()))
def test_msort_arb(v):
    result = mergesort(v)

    assert result == sorted(v)


# Merge operation tests


def test_merge_empty():
    a = []
    b = []
    result = merge(a, b)

    assert result == []


def test_merge_single_empty():
    a = [1]
    b = []
    result = merge(a, b)

    assert result == [1]

    # and the otheb way
    a = []
    b = [1]
    result = merge(a, b)

    assert result == [1]


def test_merge_singles():
    a = [1]
    b = [2]
    result = merge(a, b)

    assert result == [1, 2]

    # and the other way
    a = [2]
    b = [1]
    result = merge(a, b)

    assert result == [1, 2]


def test_merge_pairs():
    a = [1, 2]
    b = [1, 3]
    c = [2, 3]
    d = [2, 4]

    r1 = merge(a, a)
    r2 = merge(a, b)
    r3 = merge(a, c)
    r4 = merge(a, d)
    assert r1 == [1, 1, 2, 2]
    assert r2 == [1, 1, 2, 3]
    assert r3 == [1, 2, 2, 3]
    assert r4 == [1, 2, 2, 4]

    r1 = merge(b, a)
    r2 = merge(b, b)
    r3 = merge(b, c)
    r4 = merge(b, d)
    assert r1 == [1, 1, 2, 3]
    assert r2 == [1, 1, 3, 3]
    assert r3 == [1, 2, 3, 3]
    assert r4 == [1, 2, 3, 4]


@given(st.lists(st.integers()), st.lists(st.integers()))
def test_merge_list_arb(a, b):
    # inputs must be sorted
    a = sorted(a)
    b = sorted(b)

    result = merge(a, b)

    assert result == sorted(a + b)

# This isn't in Kozen but I just felt like doing something relatively easy and also it gives me an excuse to use Hypothesis.

# Given two sorted lists, merge them into a single, new sorted list
def merge(l,r):
    output = list()

    l_current = 0
    r_current = 0
    # While true because we break only when one of the lists are finished
    while True:
        # if either list is finished, append the rest of the other and return
        if len(l) == l_current:
            output.extend(r[r_current:])
            break
        if len(r) == r_current:
            output.extend(l[l_current:])
            break

        # otherwise, take the min and add it to the output
        l_item = l[l_current]
        r_item = r[r_current]
        if l_item < r_item:
            output.append(l_item)
            l_current += 1
        else:
            output.append(r_item)
            r_current += 1

    return output


# Tests
from hypothesis import given
import hypothesis.strategies as st

def test_merge_empty():
    l = []
    r = []
    result = merge(l,r)

    assert result == []

def test_merge_single_empty():
    l = [1,]
    r = []
    result = merge(l,r)

    assert result == [1,]

    # and the other way
    l = []
    r = [1,]
    result = merge(l,r)

    assert result == [1,]

def test_merge_singles():
    l = [1]
    r = [2]
    result = merge(l,r)

    assert result == [1,2]

    # and the other way
    l = [2]
    r = [1]
    result = merge(l,r)

    assert result == [1,2]

def test_merge_pairs():
    a = [1,2]
    b = [1,3]
    c = [2,3]
    d = [2,4]

    r1 = merge(a,a)
    r2 = merge(a,b)
    r3 = merge(a,c)
    r4 = merge(a,d)
    assert r1 == [1,1,2,2]
    assert r2 == [1,1,2,3]
    assert r3 == [1,2,2,3]
    assert r4 == [1,2,2,4]

    r1 = merge(b,a)
    r2 = merge(b,b)
    r3 = merge(b,c)
    r4 = merge(b,d)
    assert r1 == [1,1,2,3]
    assert r2 == [1,1,3,3]
    assert r3 == [1,2,3,3]
    assert r4 == [1,2,3,4]

@given(st.lists(st.integers()), st.lists(st.integers()))
def test_merge_list_arb(l,r):
    # inputs must be sorted
    l = sorted(l)
    r = sorted(r)

    result = merge(l,r)

    assert result == sorted(l+r)

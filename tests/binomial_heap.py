import random

from hypothesis import given
import hypothesis.strategies as st
from hypothesis.stateful import RuleBasedStateMachine, rule, precondition, invariant

from algo.binomial_heap import BinomialHeap, _BinTree, Empty

## Binomial heap tests

# Rule-based stateful testing https://hypothesis.works/articles/rule-based-stateful-testing/
# https://hypothesis.readthedocs.io/en/latest/stateful.html


class BinomialHeapMachine(RuleBasedStateMachine):
    def __init__(self):
        super().__init__()

        self.model = list()
        self.binheap = BinomialHeap()

    @rule(x=st.integers())
    def insert(self, x):
        self.model.append(x)
        self.binheap.insert(x)

    @rule()
    # precondition: the heap should not be empty => the model should not be empty
    @precondition(lambda self: self.model)
    def delete_min(self):
        model_min = min(self.model)
        self.model.remove(model_min)
        self.binheap.delete_min()

    @invariant()
    # precondition: the heap should not be empty => the model should not be empty
    @precondition(lambda self: self.model)
    def min_is_correct(self):
        model_min = min(self.model)
        heap_min = self.binheap.find_min()

        assert model_min == heap_min

    @invariant()
    def trees_in_order(self):
        """ The internal linked list of binomial trees should be strictly
        ordered by rank. """

        prev = None
        for curr in self.binheap._trees:
            if prev is None:
                prev = curr
            else:
                assert prev.rank < curr.rank


TestHeapModel = BinomialHeapMachine.TestCase

# Create and add/find_min tests


def test_binheap_create_1():
    h = BinomialHeap()
    h.insert(1)
    assert h.find_min() == 1


def test_binheap_find_min_2():
    h = BinomialHeap()
    h.insert(1)
    h.insert(0)
    assert h.find_min() == 0


def test_binheap_find_min_3():
    h = BinomialHeap()
    h.insert(3)
    h.insert(9)
    h.insert(5)
    assert h.find_min() == 3


def test_binheap_find_min_4():
    h = BinomialHeap()
    h.insert(0)
    h.insert(-1)
    assert h.find_min() == -1


def test_binheap_find_min_large_random():
    """ Hypothesis only tests smallish lists by default, which is fine because
    then it can run more tests in a reasonable amount of time. Here we test a larger list. """
    h = BinomialHeap()
    # the range is only 200, so we are guaranteed to have duplicates
    v = [random.randint(-100, 100) for _ in range(0, 10000)]

    # insert in random order
    for x in v:
        h.insert(x)
    assert min(v) == h.find_min()

    # insert in reverse-sorted order
    h = BinomialHeap()
    v_rev = sorted(v)
    for x in v_rev[::-1]:
        h.insert(x)
    assert min(v) == h.find_min()


@given(st.lists(st.integers(), min_size=1))
def test_binheap_find_min_arb(v):
    m = min(v)

    current_min = v[0]
    h = BinomialHeap()
    for x in v:
        h.insert(x)
        if x < current_min:
            current_min = x

        assert h.find_min() == current_min

    assert h.find_min() == m


# meld tests


def test_binheap_meld_1():
    h1 = BinomialHeap()
    h2 = BinomialHeap()

    h1.meld(h2)
    assert h1.find_min() is None


def test_binheap_meld_2():
    h1 = BinomialHeap()
    h1.insert(1)

    h2 = BinomialHeap()

    h1.meld(h2)
    assert h1.find_min() == 1


def test_binheap_meld_3():
    h1 = BinomialHeap()
    h1.insert(3)

    h2 = BinomialHeap()
    h2.insert(2)

    h1.meld(h2)
    assert h1.find_min() == 2


def test_binheap_meld_4():
    h1 = BinomialHeap()
    h1.insert(8)
    h1.insert(5)
    h1.insert(99)

    h2 = BinomialHeap()
    h2.insert(3)
    h2.insert(6)
    h2.insert(200)

    h1.meld(h2)
    assert h1.find_min() == 3


@given(st.lists(st.integers(), min_size=1))
def test_binheap_meld_one_empty_arb(v):
    h1 = BinomialHeap()
    h2 = BinomialHeap()

    for x in v:
        h1.insert(x)

    h1.meld(h2)
    assert h1.find_min() == min(v)

    # now with other empty

    h1 = BinomialHeap()
    h2 = BinomialHeap()

    for x in v:
        h2.insert(x)

    h1.meld(h2)
    assert h1.find_min() == min(v)


@given(st.lists(st.integers(), min_size=1), st.lists(st.integers(), min_size=1))
def test_binheap_meld_arb(v1, v2):
    h1 = BinomialHeap()
    h2 = BinomialHeap()

    for x in v1:
        h1.insert(x)
    for x in v2:
        h2.insert(x)

    h1.meld(h2)
    assert h1.find_min() == min(min(v1), min(v2))


# delete_min tests


def test_binheap_delete_min_1():
    h = BinomialHeap()
    h.insert(1)
    h.insert(2)

    h.delete_min()
    assert h.find_min() == 2

    h.delete_min()
    assert h.find_min() is None


def test_binheap_delete_min_long_fail():
    """ Do a heapsort using the binomial heap, make sure it gives the same
    answer as regular sort. """
    v = [0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, -1, 1]
    sv = sorted(v)

    h = BinomialHeap()
    for x in v:
        h.insert(x)

    for x in sv:
        assert x == h.find_min()
        h.delete_min()


def test_binheap_delete_min_long_fail_2():
    """ Do a heapsort using the binomial heap, make sure it gives the same
    answer as regular sort. """
    v = [0, 0, 0, 0, 0, 0, 0, -2, 0, -1, 0, -2, 1]
    sv = sorted(v)

    h = BinomialHeap()
    for x in v:
        h.insert(x)

    for x in sv:
        assert x == h.find_min()
        h.delete_min()


def test_binheap_delete_min_very_long():
    """ Do a heapsort using the binomial heap, make sure it gives the same
    answer as regular sort. """
    for i in range(1, 200):
        v = [0] * i + [-1, 1]
        sv = sorted(v)

        h = BinomialHeap()
        for x in v:
            h.insert(x)

        for x in sv:
            assert x == h.find_min()
            h.delete_min()


def test_binheap_delete_min_empty_err():
    h = BinomialHeap()

    try:
        h.delete_min()
    except Empty as err:
        assert "Delete min from empty heap" in str(err)

    h.insert(7)
    assert h.find_min() == 7

    h.delete_min()
    assert h.find_min() is None

    try:
        h.delete_min()
    except Empty as err:
        assert "Delete min from empty heap" in str(err)


@given(st.lists(st.integers(), min_size=1))
def test_binheap_delete_min_arb(v):
    """ Do a heapsort using the binomial heap, make sure it gives the same
    answer as regular sort. """
    sv = sorted(v)

    h = BinomialHeap()
    for x in v:
        h.insert(x)

    for x in sv:
        assert x == h.find_min()
        h.delete_min()


## Binomial tree tests

# Value tests


def test_bintree_1():
    bt = _BinTree(1)
    assert bt.rank == 0
    assert bt.value == 1


@given(st.integers())
def test_bintree_val_arb(x):
    bt = _BinTree(x)
    assert bt.rank == 0
    assert bt.value == x


# Link tests


def test_bintree_link_2():
    t1 = _BinTree(1)
    t2 = _BinTree(2)
    t1.link(t2)

    # we really shouldn't be accessing t2 anymore here as linking should be
    # a move operation. but we also want to check that merging t1 into t2 doesn't change t2.
    assert t1.rank == 1
    assert t2.rank == 0
    assert t1.value == 1

    # merge other way, still same value

    t1 = _BinTree(1)
    t2 = _BinTree(2)
    t2.link(t1)

    assert t1.rank == 0
    assert t2.rank == 1
    assert t2.value == 1


def test_bintree_link_3():
    t1 = _BinTree(1)
    t2 = _BinTree(2)
    t1.link(t2)
    assert t1.rank == 1

    t3 = _BinTree(3)
    t4 = _BinTree(4)
    t3.link(t4)
    assert t3.rank == 1
    assert t3.value == 3

    t1.link(t3)
    assert t1.rank == 2
    assert t1.value == 1

    # check some other values, still rank 2 trees

    t1 = _BinTree(4)
    t2 = _BinTree(4)
    t1.link(t2)

    t3 = _BinTree(3)
    t4 = _BinTree(2)
    t3.link(t4)

    t1.link(t3)
    assert t1.value == 2


@given(st.lists(st.integers(), min_size=8, max_size=8))
def test_bintree_link_arb_4(v):
    """ Test that we get the right min in a tree of rank 3. """
    m = min(v)
    t1 = _BinTree(v[0])
    t2 = _BinTree(v[1])
    t3 = _BinTree(v[2])
    t4 = _BinTree(v[3])

    t1.link(t2)
    t3.link(t4)
    t1.link(t3)

    t5 = _BinTree(v[4])
    t6 = _BinTree(v[5])
    t7 = _BinTree(v[6])
    t8 = _BinTree(v[7])

    t5.link(t6)
    t7.link(t8)
    t5.link(t7)

    t1.link(t5)

    assert t1.value == m


def test_bintree_link_diff_rank():
    t1 = _BinTree(4)
    t2 = _BinTree(3)
    t3 = _BinTree(2)

    t1.link(t2)  # rank 0, rank 0
    try:
        t1.link(t3)  # rank 1, rank 0 -> error
    except ValueError as err:
        assert "Cannot link binomial trees of differing ranks:" in str(err)


# pop_children tests
# FIXME: this is a bit under-tested. I actually found a bug in the
# implementation that wasn't covered by my initial test cases.
#
# The problem in writing better hypothesis tests is that it's
# annoying/slightly difficult to build arbitrary binomial trees because of the
# linking order - you have to build trees of each rank and then rank those
# together, so you're basically implementing mergesort. And then once you've
# done that, you don't actually know what any of the values of the children
# should be - so you either have to test it during the creation, or
# re-implement the code in `BinomialHeap:delete_min`.


def test_bintree_children_1():
    t1 = _BinTree(0)
    children = t1.pop_children()
    assert children.count() == 0


def test_bintree_children_2():
    t1 = _BinTree(0)
    t2 = _BinTree(1)
    t1.link(t2)

    children = t1.pop_children()
    assert children.count() == 1
    # head gives us the LL Node, .value gives us the BinTree, and .value.value
    # gives us the value of the BinTree
    assert children.head().value.value == 1

    # and the other way
    t1 = _BinTree(1)
    t2 = _BinTree(0)
    t1.link(t2)

    children = t1.pop_children()
    assert children.count() == 1
    # head gives us the LL Node, .value gives us the BinTree, and .value.value
    # gives us the value of the BinTree
    assert children.head().value.value == 1


@given(st.lists(st.integers(), min_size=8, max_size=8))
def test_bintree_children_arb_4(v):
    t1 = _BinTree(v[0])
    t2 = _BinTree(v[1])
    t3 = _BinTree(v[2])
    t4 = _BinTree(v[3])

    t1.link(t2)
    t3.link(t4)
    t1.link(t3)

    t5 = _BinTree(v[4])
    t6 = _BinTree(v[5])
    t7 = _BinTree(v[6])
    t8 = _BinTree(v[7])

    t5.link(t6)
    t7.link(t8)
    t5.link(t7)

    t1.link(t5)

    # the rank of each subtree of the root of a binomial tree increases by 1 at
    # each successive index.
    children = t1.pop_children()
    assert children.count() == 3
    for i, child in enumerate(children):
        assert child.rank == i

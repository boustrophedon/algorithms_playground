# Kozen Ch 8, p40
from typing import List, TypeVar, Optional, Any

# TODO: Use a linked list in order to do the "add with carry" operation.
# It might simplify some of the logic (it might make it more complicated
# though) and we wouldn't have to keep an index to the minimal tree, just a
# pointer.
# Additionally, it would use less memory: we wouldn't have to allocate when
# doing the traversal as we do now in the get and set methods.

# TODO: when python 3.7 gets on travis, replace 'TypeName' with TypeName in
# method signatures that use the current class's, or not-yet-defined class's, type.
#
# See https://stackoverflow.com/questions/33533148/ for why we can't use the
# actual type. this is fixed in 3.7 but travis doesn't have it.

# FIXME: constrain E to be a comparable
E = TypeVar("E", bound=Any)


class BinomialHeap:
    def __init__(self) -> None:
        """ Create an empty binomial heap. """

        self._trees: List[Optional[_BinTree]] = list()
        self._min_idx: Optional[int] = None

    def _get_min_tree(self) -> Optional["_BinTree"]:

        if self._min_idx is not None:
            min_tree = self._trees[self._min_idx]
            if min_tree:
                return min_tree
            else:
                raise RuntimeError("Minimum tree index points to an empty position.")
        else:
            return None

    def _update_min_idx(self):
        # fmt: off
        self._min_idx = min(
            # (index, minval of tree) for non-None trees
            ((i, t.value) for i, t in enumerate(self._trees) if t is not None),
            # select min by value
            key=lambda x: x[1],
            # if there are no trees in self._trees, just return None
            default=(None, None),
        )[0]  # then use the resulting index from the min tuple
        # fmt: on

    def find_min(self) -> Optional[E]:
        """ Find the minimum element of this heap. O(1) """
        t = self._get_min_tree()
        if t:
            return t.value
        else:
            return None

    def insert(self, element: E):
        """ Insert the element `element` into this heap. Amortized O(1), worst
        case O(log(n)) """

        if len(self._trees) == 0:
            self._trees.append(_BinTree(element))
            self._min_idx = 0
        else:
            new_heap = BinomialHeap()
            new_heap.insert(element)
            self.meld(new_heap)

    def delete_min(self):
        """ Delete the minimum element from this heap. O(log(n)) """

        min_tree = self._get_min_tree()
        if min_tree is None:
            # list.pop() raises IndexError, so we do the same
            raise IndexError("Delete min from empty heap")

        # get the children of the current min
        children = min_tree.pop_children()

        # remove the current min from our trees
        self._trees[self._min_idx] = None
        self._update_min_idx()

        # make a new heap from the children of the old min tree
        new_heap = BinomialHeap()
        new_heap._trees = children
        new_heap._update_min_idx()

        self.meld(new_heap)

    def meld(self, other: "BinomialHeap"):
        """ Merge heap `other` into this heap. O(log(n))

        You should not `meld(x, x)`. Additionally, you should not use
        `other` after passing it into `meld`.

        Meld works similar to doing addition by hand: you add up the trees of
        the same rank and carry if there's an extra.
        """

        us = self._trees
        them = other._trees

        carry = None

        # Getter and setters to advance by 1 along the arrays of trees.
        # They throw exceptions if the indices have fallen out of sync, at the
        # expense of allocating more memory.
        def get(l, i):
            if i < len(l):
                return l[i]
            elif i == len(l):
                l.append(None)
                return None
            else:
                raise IndexError("Can only index by 1 past the end")

        def set(l, i, x):
            if i < len(l):
                l[i] = x
            elif i == len(l):
                l.append(x)
            else:
                raise IndexError("Can only index by 1 past the end")

        # Traverse the two heaps' trees simultaneously. We use the above get
        # and set to append if we've reached the end of one of the lists.
        # Unfortunately, this means that we use extra memory to hold Nones if
        # one is significantly larger than the other.

        # The code here is very similar to adding two polynomials together, or
        # as mentioned above, doing addition by hand with carrying.

        n = max(len(us), len(them))
        for i in range(0, n):
            us_i = get(us, i)
            them_i = get(them, i)

            # if both are not none, add them and carry over, melding the
            # current carry into us if there is one.
            if us_i and them_i:
                us_i.link(them_i)
                set(us, i, None)
                set(them, i, None)
                # if there is a carry from previous, put it in the current
                if carry:
                    set(us, i, carry)
                # carry over binomial tree of rank i+1 to next iteration
                carry = us_i

            # if we have a tree and they don't, add carry if there is one,
            # otherwise keep ours.
            elif us_i and not them_i:
                if carry:
                    us_i.link(carry)
                    carry = us_i
                    set(us, i, None)
                else:  # for consistency
                    pass

            # if they have a tree and we don't, add carry if there is one,
            # otherwise move theirs to ours.
            elif not us_i and them_i:
                if carry:
                    them_i.link(carry)
                    carry = them_i
                    set(them, i, None)
                else:
                    set(us, i, them_i)

            # if neither have tree, move carry into current spot if there is one
            else:
                if carry:
                    set(us, i, carry)
                    carry = None

        # if there's a carry leftover at the end, push it
        if carry:
            us.append(carry)

        # update overall min_idx
        # TODO: we could update the min inline with the other updates, but the
        # code is messy enough already
        self._update_min_idx()


class _BinTree:
    def __init__(self, element: E) -> None:
        """ A binomial tree `B_k` with rank `k` is a tree with 2^k elements
        such that adding another `B_k` as a (rightmost) child of the root node
        makes a `B_{k+1}`.  Binomial trees of differing ranks cannot be combined.
        """

        self._value = element

        self._children: List[_BinTree] = list()

    @property
    def rank(self) -> int:
        """ Returns the rank of the binomial tree. """
        return len(self._children)

    def pop_children(self) -> List["_BinTree"]:
        """ Returns the children of the root of this tree, which should be a
        list of `_BinTree`s of increasing rank.

        You should not use this object after calling this method.
        """

        return self._children

    # See https://stackoverflow.com/questions/33533148/
    # for why we can't use the actual type. this is fixed in 3.7 but travis doesn't have it.
    def link(self, other: "_BinTree"):
        """ Combine this tree with `other` to create a new tree of rank k+1.
        The root of this tree will become the smaller of this tree or the other
        tree's root.

        You should not `link(x, x)`. Additionally, you should not use `other`
        after using it as an argument to link.
        """

        if self.rank != other.rank:
            raise ValueError(
                "Cannot link binomial trees of differing ranks: {} and {}".format(
                    self.rank, other.rank
                )
            )

        if self.value > other.value:
            self._children, other._children = other._children, self._children
            self._value, other._value = other._value, self._value

        self._children.append(other)

    @property
    def value(self) -> E:
        """ Get the value at the root of the tree """

        return self._value


### Tests
import random

from hypothesis import given
import hypothesis.strategies as st

## Binomial heap tests

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


def test_binheap_delete_min_empty_err():
    h = BinomialHeap()

    try:
        h.delete_min()
    except IndexError as err:
        assert "Delete min from empty heap" in str(err)

    h.insert(7)
    assert h.find_min() == 7

    h.delete_min()
    assert h.find_min() is None

    try:
        h.delete_min()
    except IndexError as err:
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
    assert len(children) == 0


def test_bintree_children_2():
    t1 = _BinTree(0)
    t2 = _BinTree(1)
    t1.link(t2)

    children = t1.pop_children()
    assert len(children) == 1
    assert children[0].value == 1

    # and the other way
    t1 = _BinTree(1)
    t2 = _BinTree(0)
    t1.link(t2)

    children = t1.pop_children()
    assert len(children) == 1
    assert children[0].value == 1


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
    assert len(children) == 3
    for i in range(0, 3):
        assert children[i].rank == i

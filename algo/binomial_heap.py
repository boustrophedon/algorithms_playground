# Kozen Ch 8, p40
from typing import TypeVar, Optional, Any

from algo.linked_list import LinkedList, Node

# TODO: when python 3.7 gets on travis, replace 'TypeName' with TypeName in
# method signatures that use the current class's, or not-yet-defined class's, type.
#
# See https://stackoverflow.com/questions/33533148/ for why we can't use the
# actual type. this is fixed in 3.7 but travis doesn't have it.

# FIXME: constrain E to be a comparable
E = TypeVar("E", bound=Any)


class Empty(Exception):
    pass


class BinomialHeap:
    def __init__(self) -> None:
        """ Create an empty binomial heap. """

        self._trees: LinkedList[_BinTree] = LinkedList()
        self._min_tree_node: Optional[Node[_BinTree]] = None

    def _get_min_tree(self) -> Optional["_BinTree"]:

        if self._min_tree_node:
            return self._min_tree_node.value
        else:
            return None

    def _update_min_tree_node(self):
        def min_node(min_node, curr_node):
            if curr_node.value.value < min_node.value.value:
                min_node = curr_node
            return min_node

        head = self._trees.head()
        if head is None:
            self._min_tree_node = None
        else:
            self._min_tree_node = self._trees.traverse_nodes(min_node, head)

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

        if self._trees.empty():
            self._trees.append(_BinTree(element))
            self._min_tree_node = self._trees.head()

        else:
            new_heap = BinomialHeap()
            new_heap.insert(element)
            self.meld(new_heap)

    def delete_min(self):
        """ Delete the minimum element from this heap. O(log(n)) """

        if self._min_tree_node is None:
            raise Empty("Delete min from empty heap")

        # get the children of the current min
        children = self._min_tree_node.value.pop_children()

        # remove the current min from our trees
        self._trees.remove_node(self._min_tree_node)
        self._update_min_tree_node()

        # make a new heap from the children of the old min tree
        new_heap = BinomialHeap()
        new_heap._trees = children

        self.meld(new_heap)

    def meld(self, other: "BinomialHeap"):
        """ Merge heap `other` into this heap. O(log(n))

        You should not `meld(x, x)`. Additionally, you should not use
        `other` after passing it into `meld`.
        """

        # FIXME: this code could maybe be simpler if we had a linked list that
        # kept track of the head and tail.  not necessarily a circular linked
        # list because that makes other parts of this code more complicated as
        # well.

        left = self._trees.head()
        right = other._trees.head()

        if left is None:
            self._trees = other._trees
            self._update_min_tree_node()
            return
        if right is None:
            self._update_min_tree_node()
            return

        left_tree = left.value
        right_tree = right.value

        merged: LinkedList[_BinTree] = LinkedList()
        merged_tail = None

        # merge the first two nodes outside the loop so that we can set the tail
        if left_tree.rank == right_tree.rank:
            left_tree.link(right_tree)

            merged.append(left_tree)
            merged_tail = merged.head()
            left = left.next
            right = right.next

        elif left_tree.rank < right_tree.rank:
            merged.append(left_tree)
            merged_tail = merged.head()
            left = left.next
        else:
            merged.append(right_tree)
            merged_tail = merged.head()
            right = right.next

        while left is not None and right is not None:
            # The type checker doesn't know that merged_tail cannot be none by
            # consturction, so we have to assert.
            assert merged_tail is not None

            left_tree = left.value
            right_tree = right.value
            merged_tree = merged_tail.value

            # If both trees have the same rank, link them and add it to the end
            # of the merged list, further linking it with tree at the tail of the
            # merged list if they also have the same rank.
            if left_tree.rank == right_tree.rank:
                left_tree.link(right_tree)

                if left_tree.rank == merged_tree.rank:
                    merged_tree.link(left_tree)
                else:
                    merged_tail.next = Node(left_tree, None)
                    merged_tail = merged_tail.next

                left = left.next
                right = right.next

            # Otherwise, add the smaller-ranked tree to the merged list,
            # linking it with the tree at the tail of the merged list if they
            # have the same rank.
            elif left_tree.rank < right_tree.rank:
                merged_tail.next = Node(left_tree, None)
                merged_tail = merged_tail.next
                left = left.next

            else:
                merged_tail.next = Node(right_tree, None)
                merged_tail = merged_tail.next
                right = right.next

        # If there are leftovers, append them to the rest of the tree, linking
        # if necessary.
        leftover = None
        if left is not None:
            leftover = left
        elif right is not None:
            leftover = right

        while leftover is not None:
            # The type checker doesn't know that merged_tail cannot be None by
            # consturction, so we have to assert.
            assert merged_tail is not None

            tree = leftover.value
            merged_tree = merged_tail.value
            if tree.rank == merged_tree.rank:
                merged_tree.link(tree)
            else:
                merged_tail.next = Node(tree, None)
                merged_tail = merged_tail.next
            leftover = leftover.next

        self._trees = merged
        self._update_min_tree_node()


class _BinTree:
    def __init__(self, element: E) -> None:
        """ A binomial tree `B_k` with rank `k` is a tree with 2^k elements
        such that adding another `B_k` as a (rightmost) child of the root node
        makes a `B_{k+1}`.  Binomial trees of differing ranks cannot be combined.
        """

        self._value: E = element
        self._children: LinkedList[_BinTree] = LinkedList()
        self._rank: int = 0

    @property
    def rank(self) -> int:
        """ Returns the rank of the binomial tree. """
        return self._rank

    def pop_children(self) -> LinkedList["_BinTree"]:
        """ Returns the children of the root of this tree, which should be a
        linked list of `_BinTree`s of increasing rank.

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

        This operation is O(k).
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
        self._rank += 1

    @property
    def value(self) -> E:
        """ Get the value at the root of the tree """

        return self._value

    def __str__(self):
        return f"rank {self.rank}, val {self.value}"


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

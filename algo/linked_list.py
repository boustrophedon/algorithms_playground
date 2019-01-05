from typing import TypeVar, Optional, Generic, Callable

E = TypeVar("E")
Ctx = TypeVar("Ctx")


class Node(Generic[E]):
    """ A node in a linked list """

    def __init__(self, value: E, next: Optional["Node"]):
        self.value = value
        self.next = next


class LinkedList(Generic[E]):
    """ A singly-linked list. """

    def __init__(self):
        self._head: Optional[Node[E]] = None

    def head(self) -> Optional[Node[E]]:
        """ Returns the head node of the linked list. """
        return self._head

    def append(self, e: E):
        """ Appends the element `e` to the end of the linked list. O(n) time.
        """
        new_node = Node(e, None)
        if self._head is None:
            self._head = new_node
            return

        curr = self._head
        while curr.next is not None:
            curr = curr.next
        curr.next = new_node

    def prepend(self, e: E):
        """ Prepends the element `e` to the front of the linked list, making it
        the new head. O(1) time.
        """
        raise NotImplementedError

    def pop(self) -> E:
        """ Returns the tail element of the linked list. Throws an exception if
        the linked list is empty. O(n) time. """
        raise NotImplementedError

    def popleft(self) -> E:
        """ Returns the head element of the linked list. Throws an exception if
        the linked list is empty. O(1) time. """
        raise NotImplementedError

    # FIXME: should I use a context parameter or just `...`, which is apparently legal
    # See https://docs.python.org/3/library/typing.html#callable
    def traverse(self, f: Callable[[Ctx, E], Ctx], initial_context: Ctx) -> Ctx:
        """
        Call f(ctx, e) -> ctx on every value of the nodes of the linked list.
        Traversal ends when either the function raises a StopIteration
        or when we reach the last node of the list.

        That is, the return type of `f` is a context object that gets updated each
        iteration starting with `initial_context`, and the return type of
        `traverse` is the resulting final ctx object when iteration has ended.

        If `StopIteration` is used to return early from the iteration, the
        `value` attribute on the StopIteration exception will be used as the
        return value.

        We return the ctx from `f` so that it's easier to use with
        immutable objects, e.g. ints and strings. Otherwise, to implement, e.g.
        count() as below, you'd have to wrap them in a container.

        O(n*O(f)) time.
        """
        ctx = initial_context
        curr = self._head

        while curr is not None:
            try:
                ctx = f(ctx, curr.value)
            except StopIteration as end:
                return end.value
            curr = curr.next

        return ctx

    def count(self) -> int:
        """ Returns the number of items in the linked list. O(n) time. """
        raise NotImplementedError

    def pprint(self):
        """ Pretty print the values of the elements of the linked list,
        separated by arrows.
        """
        raise NotImplementedError


from hypothesis import given
import hypothesis.strategies as st

### Linked List tests


## Traverse tests

# Traverse test helper functions
def empty(ctx, x):
    None


def assert_on_call(ctx, x):
    assert False, "should not be called"


def count_items(ctx, x):
    return ctx + 1


class CounterStop:
    def __init__(self, stop_at):
        self.stop_at = stop_at
        self.count = 0


def end_at_item(ctx, x):
    """ Stop at the item equal to x, or at the last item if none are equal to
    x. """
    ctx.count += 1
    if ctx.stop_at == x:
        raise StopIteration(ctx)
    else:
        return ctx


# make sure f is not called if the linked list is empty, but we get back the
# context
def test_ll_traverse_empty():
    ll = LinkedList()
    result = ll.traverse(assert_on_call, "test")
    assert result == "test"


def test_ll_traverse_hits_every_element():
    ll = LinkedList()
    # check with one element
    ll.append("a")
    result = ll.traverse(count_items, 0)
    assert result == 1

    # check with two elements
    ll.append("b")
    result = ll.traverse(count_items, 0)
    assert result == 2

    # check with three elements
    ll.append("c")
    result = ll.traverse(count_items, 0)
    assert result == 3


def test_ll_traverse_stop_iteration():
    ll = LinkedList()
    ll.append("a")
    ll.append("b")
    ll.append("c")
    ll.append("d")

    # stop at first
    result = ll.traverse(end_at_item, CounterStop("a"))
    assert result.count == 1

    # stop at second
    result = ll.traverse(end_at_item, CounterStop("b"))
    assert result.count == 2

    # stop at third
    result = ll.traverse(end_at_item, CounterStop("c"))
    assert result.count == 3

    # stop element not in list
    result = ll.traverse(end_at_item, CounterStop("x"))
    assert result.count == 4

    # add new element and stop element still not in list, but count increases
    ll.append("e")
    result = ll.traverse(end_at_item, CounterStop("x"))
    assert result.count == 5


@given(st.lists(st.integers()))
def test_ll_traverse_hits_every_element_arb(v):
    ll = LinkedList()
    for i, x in enumerate(v):
        ll.append(x)
        result = ll.traverse(count_items, 0)
        # i+1 is the number of elements since enumerate starts at 0
        assert result == i + 1

    # finally, check again (even though it's the same as the last iteration in
    # the above loop) that the total number of items is the same
    result = ll.traverse(count_items, 0)
    assert result == len(v)


# set 1 is all negative numbers, and set 2 is all positive numbers, so they can't intersect
# it really doesn't matter what the elements are for this test
#
# Note that they are sets, because end_at_item stops at the first item, which
# may not otherwise be unique
@given(st.sets(st.integers(max_value=0)), st.sets(st.integers(min_value=1)))
def test_ll_traverse_stop_iteration_arb(v1, v2):
    ll = LinkedList()
    for x in v1:
        ll.append(x)

    # Count the number of elements we traverse
    for i, x in enumerate(v1):
        # stop at x
        result = ll.traverse(end_at_item, CounterStop(x))
        # i+1 is the number of elements since enumerate starts at 0
        assert result.count == i + 1

    # now do it in reverse for good measure
    for i, x in reversed(list(enumerate(v1))):
        # stop at x
        result = ll.traverse(end_at_item, CounterStop(x))
        # i+1 is the number of elements since enumerate starts at 0
        assert result.count == i + 1

    # Now since every element of v2 isn't in v1 and subsequently the list
    # Check that we traverse the whole list when searching for them
    for x in v2:
        result = ll.traverse(end_at_item, CounterStop(x))

        # v1 items inserted into list
        assert result.count == len(v1)


## Append tests
# Test that head is properly updated when appending to empty list
def test_ll_append_1():
    ll = LinkedList()
    ll.append(1)

    assert ll.head().value == 1
    assert ll.head().next is None


def test_ll_append_2():
    ll = LinkedList()
    ll.append(1)
    ll.append(2)

    assert ll.head().value == 1
    assert ll.head().next is not None


# append elements of a list to LinkedList, then check they are there in the
# same order as the original
@given(st.lists(st.integers()))
def test_ll_append_arb(v):
    ll = LinkedList()
    for x in v:
        ll.append(x)

    curr = ll.head()
    for i in range(0, len(v)):
        assert curr.value == v[i]
        curr = curr.next

    # after going through all the elements, check we've hit None at the end
    assert curr is None


## Misc tests
# Head is empty at start
def test_ll_head_empty():
    ll = LinkedList()
    assert ll.head() is None


### Node tests (very brief, it just holds data)
def test_node_1():
    n = Node(1, None)
    assert n.value == 1
    assert n.next is None


def test_node_2():
    n1 = Node(1, None)
    n2 = Node(2, n1)
    assert n2.value == 2
    assert n2.next == n1
    assert n2.next.value == n1.value == 1
    assert n2.next.next == n1.next is None

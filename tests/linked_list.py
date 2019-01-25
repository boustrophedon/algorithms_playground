from collections import deque

from hypothesis import given
import hypothesis.strategies as st
from hypothesis.stateful import RuleBasedStateMachine, rule, precondition, invariant

from algo.linked_list import LinkedList, Node, Empty


## Model-based testing


class LinkedListModel(RuleBasedStateMachine):
    def __init__(self):
        super().__init__()

        self.model = deque()
        self.ll = LinkedList()

    @invariant()
    def same_items_in_same_order(self):
        assert len(self.model) == self.ll.count()
        for (m_item, l_item) in zip(self.model, self.ll):
            assert m_item == l_item

    @rule(x=st.integers())
    def prepend(self, x):
        self.model.appendleft(x)
        self.ll.prepend(x)

    @rule(x=st.integers())
    def append(self, x):
        self.model.append(x)
        self.ll.append(x)

    @rule()
    # model/list should not be empty when popping
    @precondition(lambda self: self.model)
    def popleft(self):
        self.model.popleft()
        self.ll.popleft()

    @rule()
    # model/list should not be empty when popping
    @precondition(lambda self: self.model)
    def pop(self):
        self.model.pop()
        self.ll.pop()


TestLLModel = LinkedListModel.TestCase

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


## remove tests

import random


def test_ll_remove_empty():
    ll = LinkedList()
    n = Node(1, None)
    try:
        ll.remove_node(n)
    except Empty:
        pass
    except Exception as err:
        assert False, f"Other exception thrown inside iterator: {type(err)}"
    else:
        assert False, "remove call should have thrown an exception but it did not"


@given(st.lists(st.integers(), min_size=1))
def test_ll_remove_not_found_arb(v):
    ll = LinkedList()
    for x in v:
        ll.prepend(x)

    # contents doesn't matter because remove removes the *node*, not the item,
    # and nodes with equal values are not the same node.
    n = Node(1, None)
    try:
        ll.remove_node(n)
    except ValueError:
        pass
    except Exception as err:
        assert False, f"Other exception thrown inside iterator: {type(err)}"
    else:
        assert False, "remove call should have thrown an exception but it did not"


def test_ll_remove_head_only():
    ll = LinkedList()
    ll.append(1)
    head = ll.head()
    ll.remove_node(head)

    assert ll.count() == 0
    assert ll.head() is None


def test_ll_remove_head_1():
    ll = LinkedList()
    ll.append(1)
    ll.append(2)
    head = ll.head()
    ll.remove_node(head)

    assert ll.count() == 1
    assert ll.head().value == 2


def test_ll_remove_tail_1():
    ll = LinkedList()
    ll.append(1)
    ll.prepend(2)
    head = ll.head()
    ll.remove_node(head)

    assert ll.count() == 1
    assert ll.head().value == 1


def test_ll_remove_middle():
    ll = LinkedList()
    ll.append(1)
    ll.append(2)
    ll.append(3)
    middle = ll.head().next
    ll.remove_node(middle)

    assert ll.count() == 2
    assert ll.head().value == 1
    assert ll.head().next.value == 3


@given(st.lists(st.integers()))
def test_ll_remove_in_order_arb(v):
    ll = LinkedList()

    for x in v:
        # prepending is faster
        ll.prepend(x)

    # since we prepended, ll items are in reverse order
    for x in reversed(v):
        assert ll.head().value == x
        ll.remove_node(ll.head())

    assert ll.count() == 0
    assert ll.head() is None


@given(st.lists(st.integers()))
def test_ll_remove_random_order_arb(v):
    ll = LinkedList()

    for x in v:
        ll.prepend(x)

    def collect_nodes(nodes_set, curr):
        nodes_set.add(curr)
        return nodes_set

    nodes_set = ll.traverse_nodes(collect_nodes, set())
    shuffled_nodes = list(nodes_set)
    random.shuffle(shuffled_nodes)

    for node in shuffled_nodes:
        nodes_set.remove(node)
        ll.remove_node(node)
        assert nodes_set == ll.traverse_nodes(collect_nodes, set())


## __bool__ tests
def test_ll_bool_empty():
    ll = LinkedList()
    if ll:
        assert False, "Returned True in if statement even though the list was empty."
    else:
        assert True


def test_ll_bool_nonempty():
    ll = LinkedList()
    ll.append("a")
    if ll:
        assert True
    else:
        assert (
            False
        ), "Returned False in if statement even though the list was not empty."

    # now remove element and check it returns False again
    ll.pop()
    if ll:
        assert False, "Returned True in if statement even though the list was empty."
    else:
        assert True


## Count tests
def test_ll_count_empty():
    ll = LinkedList()
    assert ll.count() == 0


def test_ll_count_1():
    ll = LinkedList()
    ll.append("hi")
    assert ll.count() == 1


def test_ll_count_2():
    ll = LinkedList()
    ll.append("hello")
    ll.append("there")
    assert ll.count() == 2


@given(st.lists(st.integers()))
def test_ll_count_arb(v):
    ll = LinkedList()

    for x in v:
        ll.prepend(x)
    assert ll.count() == len(v)


## Pretty print tests

from io import StringIO


def test_ll_pprint_empty():
    ll = LinkedList()
    f = StringIO()

    ll.pprint(file=f)
    assert f.getvalue() == "\n"


def test_ll_pprint_single_value():
    ll = LinkedList()
    f = StringIO()

    ll.append(1)
    ll.pprint(file=f)
    assert f.getvalue() == "1\n"


def test_ll_pprint_3():
    ll = LinkedList()
    f = StringIO()

    ll.append(2)
    ll.append(1)
    ll.append(3)

    ll.pprint(file=f)
    assert f.getvalue() == "2 -> 1 -> 3\n"


@given(st.lists(st.integers(), min_size=2))
def test_ll_pprint_arb(v):
    ll = LinkedList()
    f = StringIO()
    for x in v:
        ll.append(x)

    ll.pprint(file=f)
    contents = f.getvalue()
    for x in v:
        assert str(x) in contents

    assert contents.count("->") == len(v) - 1


## Iterator tests
def test_ll_iterator_empty():
    ll = LinkedList()
    for x in ll:
        assert False, "Inside iterator but no elements in list"


def test_ll_iterator_single():
    ll = LinkedList()

    ll.append("test")
    it = iter(ll)
    assert next(it) == "test"
    try:
        next(it)
        assert False, "Above line should throw exception."
    except StopIteration:
        pass
    except Exception as err:
        assert False, f"Other exception thrown inside iterator: {type(err)}"
    else:
        assert False, "No exception thrown at end of iteration"


def test_ll_iterator_2():
    ll = LinkedList()

    ll.append("test1")
    ll.append("test2")
    it = iter(ll)

    assert next(it) == "test1"
    assert next(it) == "test2"
    try:
        next(it)
        assert False, "Above line should throw exception."
    except StopIteration:
        pass
    except Exception as err:
        assert False, f"Other exception thrown inside iterator: {type(err)}"


@given(st.lists(st.integers()))
def test_ll_iterator_arb(v):
    ll = LinkedList()

    for x in v:
        ll.append(x)

    for x_ll, x_v in zip(ll, v):
        assert x_ll == x_v


## Pop tests
def test_ll_pop_empty():
    ll = LinkedList()
    try:
        ll.pop()
        assert False, "Above line should throw exception."
    except Empty:
        pass  # We should get empty
    except Exception as err:
        assert (
            False
        ), f"Non-empty exception raised when popping from empty list: {type(err)}"


def test_ll_pop_1():
    ll = LinkedList()
    ll.append(1)
    assert ll.pop() == 1


def test_ll_pop_2():
    ll = LinkedList()
    ll.append(1)
    ll.append(2)
    assert ll.pop() == 2
    assert ll.pop() == 1


# TODO: use hypothesis rule-based model testing
@given(st.lists(st.integers()))
def test_ll_pop_in_order_arb(v):
    ll = LinkedList()
    for x in v:
        ll.append(x)

    for x in reversed(v):
        assert ll.pop() == x

    # make sure that we get empty exception once we've popped everything
    try:
        ll.pop()
        assert False, "Above line should throw exception."
    except Empty:
        pass  # We should get empty
    except Exception as err:
        assert (
            False
        ), f"Non-empty exception raised when popping from empty list: {type(err)}"


@given(st.lists(st.integers()), st.lists(st.integers()))
def test_ll_pop_interspersed_arb(v1, v2):
    """ Insert all elements of v1, then alternate between popping and inserting
    elements of v2 and v1. """
    ll = LinkedList()
    for x in v1:
        ll.append(x)
    for x in reversed(v1):
        assert ll.pop() == x

        # now add element of v2 and check popping gives the correct element
        if len(v2) > 0:
            x2 = v2.pop()
            ll.append(x2)
            assert ll.pop() == x2

    # make sure that we get empty exception once we've popped everything
    try:
        ll.pop()
        assert False, "Above line should throw exception."
    except Empty:
        pass  # We should get empty
    except Exception as err:
        assert (
            False
        ), f"Non-empty exception raised when popping from empty list: {type(err)}"


## Popleft tests
def test_ll_popleft_empty():
    ll = LinkedList()
    try:
        ll.popleft()
        assert False, "Above line should throw exception."
    except Empty:
        pass  # We should get empty
    except Exception as err:
        assert (
            False
        ), f"Non-empty exception raised when popleft from empty list: {type(err)}"


def test_ll_popleft_1():
    ll = LinkedList()
    ll.append(1)
    assert ll.popleft() == 1


def test_ll_popleft_2():
    ll = LinkedList()
    ll.append(1)
    ll.append(2)
    assert ll.popleft() == 1
    assert ll.popleft() == 2


# TODO: use hypothesis rule-based model testing
@given(st.lists(st.integers()))
def test_ll_popleft_in_order_arb(v):
    ll = LinkedList()
    for x in v:
        ll.append(x)

    for x in v:
        assert ll.popleft() == x

    # make sure that we get empty exception once we've popped everything
    try:
        ll.popleft()
        assert False, "Above line should throw exception."
    except Empty:
        pass  # We should get empty
    except Exception as err:
        assert (
            False
        ), f"Non-empty exception raised when popleft from empty list: {type(err)}"


@given(st.lists(st.integers()), st.lists(st.integers()))
def test_ll_popleft_interspersed_arb(v1, v2):
    """ Insert all elements of v1, then alternate between popleft and
    prepending elements of v2 and v1. """
    ll = LinkedList()
    for x in v1:
        ll.append(x)
    for x in v1:
        assert ll.popleft() == x

        # now add element of v2 and check popleftping gives the correct element
        if len(v2) > 0:
            x2 = v2.pop(0)  # equivalent of popleft
            ll.prepend(x2)
            assert ll.popleft() == x2

    # make sure that we get empty exception once we've popped everything
    try:
        ll.popleft()
        assert False, "Above line should throw exception."
    except Empty:
        pass  # We should get empty
    except Exception as err:
        assert (
            False
        ), f"Non-empty exception raised when popleft from empty list: {type(err)}"


## Prepend tests
# Test that head is properly updated when prepending to empty list
def test_ll_prepend_1():
    ll = LinkedList()
    ll.prepend(1)

    assert ll.head().value == 1
    assert ll.head().next is None


def test_ll_prepend_2():
    ll = LinkedList()
    ll.prepend(1)
    ll.prepend(2)

    assert ll.head().value == 2
    assert ll.head().next is not None

    assert ll.head().next.value == 1
    assert ll.head().next.next is None


# prepend elements of a list to LinkedList, then check they are there in the
# reverse order as the original
@given(st.lists(st.integers()))
def test_ll_prepend_arb(v):
    ll = LinkedList()
    for x in v:
        ll.prepend(x)

    curr = ll.head()
    for i in reversed(range(0, len(v))):
        assert curr.value == v[i]
        curr = curr.next

    # after going through all the elements, check we've hit None at the end
    assert curr is None


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

    assert ll.head().next.value == 2
    assert ll.head().next.next is None


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

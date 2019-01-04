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
        raise NotImplementedError

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

# Head is empty at start
def test_ll_head_empty():
    ll = LinkedList()
    assert ll.head() is None


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

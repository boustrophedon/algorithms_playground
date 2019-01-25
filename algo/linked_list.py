from typing import List, TypeVar, Optional, Generic, Callable

E = TypeVar("E")
Ctx = TypeVar("Ctx")


class Empty(Exception):
    """ Exception raised during an operation when the linked list is empty """

    pass


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

    def empty(self) -> bool:
        """ Returns True if the linked list has no elements, False otherwise. O(1) time. """
        if self._head:
            return False
        else:
            return True

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
        new_node = Node(e, None)
        if self._head is None:
            self._head = new_node
            return

        # attach the new head on top of the current one
        new_node.next = self._head
        # set the head to the new node
        self._head = new_node

    def pop(self) -> E:
        """ Returns and removes the tail element of the linked list. Throws an
        `Empty` exception if the linked list is empty. O(n) time. """
        if self._head is None:
            raise Empty

        # special case for popping the head
        if self._head.next is None:
            result = self._head.value
            self._head = None
            return result

        # FIXME: traverse should probably have two type parameters, one for ctx
        # and one for output so it's a fold_map/filter_map kind of thing
        def second_to_last(_n: Node[E], curr: Node[E]) -> Node[E]:
            # there's no way to do an unwrap and we know that curr.next.next will always be valid
            if curr.next.next is None:  # type: ignore
                raise StopIteration(curr)

            # this line will never be reached
            return _n

        # self._head is unused
        new_last = self.traverse_nodes(second_to_last, self._head)
        # get the last item's value
        result = new_last.next.value  # type: ignore
        # remove last from the list
        new_last.next = None

        return result

    def popleft(self) -> E:
        """ Returns and removes the head element of the linked list. Throws an
        `Empty` exception if the linked list is empty. O(1) time. """
        if self._head is None:
            raise Empty()

        # special case for removing the last node
        if self._head.next is None:
            result = self._head.value
            self._head = None
            return result

        result = self._head.value
        self._head = self._head.next

        return result

    def remove_node(self, node: Node[E]):
        """ Remove the Node `node` from the list, connecting its previous node to its next.

        Raises a ValueError if `node` is not in the list. Raise Empty if list is empty.

        This is an O(n) operation because this list is singly-linked.
        """

        if self._head is None:
            raise Empty()

        # special case for removing the head node
        if self._head is node:
            self._head = self._head.next
            return

        curr = self._head
        while curr is not None:
            if curr.next is node:
                curr.next = curr.next.next  # type: ignore
                return
            curr = curr.next  # type: ignore

        raise ValueError(f"Node {node} not found in linked list")

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

        This function is similar to a fold or reduce, but it doesn't
        necessarily consume the elements of the list and you can stop early.

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

    # I didn't bother writing tests for this as it's essentially the same code as traverse above
    # FIXME: technically I should just write traverse in terms of
    # traverse_nodes by unwrapping the node's value.
    def traverse_nodes(
        self, f: Callable[[Ctx, Node[E]], Ctx], initial_context: Ctx
    ) -> Ctx:
        """
        Call f(ctx, e) -> ctx on every node of the linked list.
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

        This function is similar to a fold or reduce, but it doesn't
        necessarily consume the elements of the list and you can stop early.

        O(n*O(f)) time.
        """
        ctx = initial_context
        curr = self._head

        while curr is not None:
            try:
                ctx = f(ctx, curr)
            except StopIteration as end:
                return end.value
            curr = curr.next

        return ctx

    def count(self) -> int:
        """ Returns the number of items in the linked list. O(n) time. """
        count = 0
        for _ in self:
            count += 1
        return count

    def pprint(self, file=None):
        """ Pretty print the values of the elements of the linked list,
        separated by arrows. O(n)
        """

        def print_node(buf: List[str], curr: Node[E]) -> List[str]:
            buf.append(str(curr.value))
            if curr.next is not None:
                buf.append("->")
            return buf

        print(" ".join(self.traverse_nodes(print_node, list())), file=file)

    def __iter__(self):
        if self._head is None:
            return

        curr = self._head
        while curr is not None:
            yield curr.value
            curr = curr.next

    def __bool__(self):
        """ Returns True if the list has at least one element, and False otherwise.
        Used for `if ll: ...` constructions.
        """
        return not self.empty()

    # for python 2 compatibility
    # __nonzero__ == __bool__

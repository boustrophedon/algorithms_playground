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

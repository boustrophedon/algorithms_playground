from typing import Iterable, Hashable, Tuple, Any, MutableMapping, Dict

from hypothesis import given
from hypothesis import strategies as st


def check_maps_equal(
    items: Iterable[Tuple[Hashable, Any]], test_map: MutableMapping
) -> Tuple[bool, str]:
    expected = dict()

    # insert everything into both maps
    for k, v in items:
        expected[k] = v
        test_map[k] = v

    # assert both maps have the same number of items
    if len(expected) != len(test_map):
        return (False, "Maps do not have same number of items")

    # assert that all keys in expected match test
    for k, v in expected.items():
        if k not in test_map:
            return (False, "Key {} missing from test map".format(k))
        if test_map[k] != v:
            return (
                False,
                "Key {} does not have the correct value {} in test map".format(k, v),
            )

    return (True, "Tests passed")


def assert_maps_equal(items: Iterable[Tuple[Hashable, Any]], test_map: MutableMapping):
    result = check_maps_equal(items, test_map)
    assert result[0], result[1]


class BadDict(MutableMapping):
    """A dict that returns contradictory results. It throws a KeyError for
    every input but claims to have length 1, yet returns an empty list from
    `__iter__`."""

    def __init__(self, iterable=None):
        pass

    def __delitem__(self):
        pass

    def __getitem__(self, k):
        raise KeyError(k)

    def __iter__(self):
        return list()

    def __len__(self):
        return 1

    def __setitem__(self, k, v):
        pass


@given(st.dictionaries(st.integers(), st.text()))
def test_maps_equal(d: Dict[int, str]):
    items = list(d.items())

    assert_maps_equal(items, dict())


@given(st.dictionaries(st.integers(), st.text()))
def test_maps_not_equal(d: Dict[int, str]):
    items = d.items()

    result = check_maps_equal(items, BadDict())

    assert result[0] == False, "Test passed with BadDict"

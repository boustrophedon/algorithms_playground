from algo.strassen import strassen, get_blocks

import numpy as np

## Test 2x2 base case


def test_2x2_zero():
    m1 = np.zeros((2, 2), dtype=int)
    m2 = np.zeros((2, 2), dtype=int)

    strassen_result = strassen(m1, m2)
    numpy_result = m1 @ m2

    np.testing.assert_equal(strassen_result, numpy_result)


def test_2x2_rand():
    m1 = np.random.randint(-100, high=100, size=(2, 2))
    m2 = np.random.randint(-100, high=100, size=(2, 2))

    strassen_result = strassen(m1, m2)
    numpy_result = m1 @ m2

    np.testing.assert_equal(strassen_result, numpy_result)


## Test 4x4 recursive case


def test_4x4_ones():
    m1 = np.ones((4, 4))
    m2 = np.ones((4, 4))

    strassen_result = strassen(m1, m2)
    numpy_result = m1 @ m2

    np.testing.assert_equal(strassen_result, numpy_result)


def test_4x4_increasing_blocks():
    m1 = np.eye((4))
    m2 = np.block(
        [
            [np.ones((2, 2)), 2 * np.ones((2, 2))],
            [3 * np.ones((2, 2)), 4 * np.ones((2, 2))],
        ]
    )

    strassen_result = strassen(m1, m2)
    numpy_result = m1 @ m2

    np.testing.assert_equal(strassen_result, numpy_result)

    # swap order

    strassen_result = strassen(m2, m1)
    numpy_result = m2 @ m1

    np.testing.assert_equal(strassen_result, numpy_result)

    # block with block

    strassen_result = strassen(m2, m2)
    numpy_result = m2 @ m2

    np.testing.assert_equal(strassen_result, numpy_result)


def test_4x4_range():
    m1 = np.reshape(range(0, 16), (4, 4))
    m2 = np.eye(4)

    strassen_result = strassen(m1, m2)
    numpy_result = m1 @ m2

    np.testing.assert_equal(strassen_result, numpy_result)

    # swap order

    strassen_result = strassen(m2, m1)
    numpy_result = m2 @ m1

    np.testing.assert_equal(strassen_result, numpy_result)

    # range with range

    strassen_result = strassen(m1, m1)
    numpy_result = m1 @ m1

    np.testing.assert_equal(strassen_result, numpy_result)


def test_4x4_rand():
    m1 = np.random.randint(-10, high=10, size=(4, 4))
    m2 = np.random.randint(-10, high=10, size=(4, 4))

    strassen_result = strassen(m1, m2)
    numpy_result = m1 @ m2

    np.testing.assert_equal(strassen_result, numpy_result)


## test blocks


def test_blocks_4x4_1():
    blocks = [np.eye(2), np.zeros((2, 2)), np.zeros((2, 2)), np.eye(2)]
    m = np.eye((4))

    for i, t in enumerate(zip(blocks, get_blocks(m))):
        np.testing.assert_equal(t[0], t[1])


def test_blocks_4x4_2():
    blocks = [np.eye(2), 2 * np.eye(2), 3 * np.eye(2), 4 * np.eye(2)]
    m = np.block([[blocks[0], blocks[1]], [blocks[2], blocks[3]]])

    for i, t in enumerate(zip(blocks, get_blocks(m))):
        np.testing.assert_equal(t[0], t[1])


def test_blocks_4x4_rand():
    blocks = [np.random.randint(-100, high=100, size=(2, 2)) for _ in range(0, 4)]
    m = np.block([[blocks[0], blocks[1]], [blocks[2], blocks[3]]])

    for i, t in enumerate(zip(blocks, get_blocks(m))):
        np.testing.assert_equal(t[0], t[1])

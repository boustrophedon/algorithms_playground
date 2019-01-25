# Implements the Strassen matrix multiplication algorithm strictly for square
# matrices with power-of-2 dimensions. Page 7, section 1.4 of Kozen.

# Note that the formulae in the book are subtly incorrect: S_5 and S_7 should be reversed

from typing import List
import numpy as np


def get_blocks(m: np.ndarray) -> List[np.ndarray]:
    """ Split a 2^n x 2^n matrix into 4 blocks of size 2^(n-1) by 2^(n-1)"""
    n = m.shape[0]
    n2 = n // 2

    a = m[0:n2, 0:n2]
    b = m[0:n2, n2:n]
    c = m[n2:n, 0:n2]
    d = m[n2:n, n2:n]

    blocks = [a, b, c, d]
    return blocks


def strassen(m1: np.ndarray, m2: np.ndarray) -> np.array:
    """ Multiply matrices m1 and m2 using the Strassen algorithm. m1 and m2
    must be square, power-of-2-sized matrices. """

    # assert matrices have same size and are square
    assert m1.shape == m2.shape
    assert m1.shape[0] == m1.shape[1]
    # assert matrix dimension is power of 2
    # by checking only one bit is set in binary representation
    assert bin(m1.shape[0]).count("1") == 1

    # if we are 2x2 matrix, operate directly on elements
    if m1.shape[0] == 2:
        a = m1[0, 0]
        b = m1[0, 1]
        c = m1[1, 0]
        d = m1[1, 1]

        e = m2[0, 0]
        f = m2[0, 1]
        g = m2[1, 0]
        h = m2[1, 1]

        s1 = (b - d) * (g + h)
        s2 = (a + d) * (e + h)
        s3 = (a - c) * (e + f)
        s4 = h * (a + b)
        s5 = (f - h) * a
        s6 = d * (g - e)
        s7 = (c + d) * e

        n11 = s1 + s2 - s4 + s6
        n12 = s4 + s5
        n21 = s6 + s7
        n22 = s2 - s3 + s5 - s7
        return np.array([[n11, n12], [n21, n22]])

    # else break into block matrices and run recursively
    else:
        a, b, c, d = get_blocks(m1)
        e, f, g, h = get_blocks(m2)

        s1 = strassen((b - d), (g + h))
        np.testing.assert_equal(s1, (b - d) @ (g + h))
        s2 = strassen((a + d), (e + h))
        np.testing.assert_equal(s2, (a + d) @ (e + h))
        s3 = strassen((a - c), (e + f))
        np.testing.assert_equal(s3, (a - c) @ (e + f))
        s4 = strassen((a + b), h)
        np.testing.assert_equal(s4, (a + b) @ h)
        s5 = strassen(a, (f - h))
        np.testing.assert_equal(s5, a @ (f - h))
        s6 = strassen(d, (g - e))
        np.testing.assert_equal(s6, d @ (g - e))
        s7 = strassen((c + d), e)
        np.testing.assert_equal(s7, (c + d) @ e)

        n11 = s1 + s2 - s4 + s6
        n12 = s4 + s5
        n21 = s6 + s7
        n22 = s2 - s3 + s5 - s7

        return np.block([[n11, n12], [n21, n22]])

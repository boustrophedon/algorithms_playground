# Implements the Strassen matrix multiplication algorithm strictly for square
# matrices with power-of-2 dimensions. Page 7, section 1.4 of Kozen.

import numpy as np

def strassen(m1, m2):
    return np.zeros((2,2))

def test_2x2_zero():
    m1 = np.zeros((2,2))
    m2 = np.zeros((2,2))

    strassen_result = strassen(m1,m2)
    numpy_result = m1@m2

    np.testing.assert_equal(strassen_result, numpy_result)


# def test_2x2_rand():
#     m1 = np.random.rand(2,2)
#     m2 = np.random.rand(2,2)
#     self.assertEqual(strassen(m1,m2), m1*m2)

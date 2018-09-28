# Implements the Strassen matrix multiplication algorithm strictly for square
# matrices with power-of-2 dimensions. Page 7, section 1.4 of Kozen.

# Note that the formulae in the book are subtly incorrect: S_5 and S_7 should be reversed 

import numpy as np

def get_blocks(m):
    n = m.shape[0]
    n2 = n//2

    a = m[0:n2, 0:n2]
    b = m[0:n2, n2:n]
    c = m[n2:n, 0:n2]
    d = m[n2:n, n2:n]

    blocks = [a,b,c,d]
    return blocks

def strassen(m1, m2):
    # assert matrices have same size and are square
    assert m1.shape == m2.shape
    assert m1.shape[0] == m1.shape[1]
    # assert matrix dimension is power of 2
    # by checking only one bit is set in binary representation
    assert bin(m1.shape[0]).count("1") == 1

    # if we are 2x2 matrix, operate directly on elements
    if m1.shape[0] == 2:
        a = m1[0,0]
        b = m1[0,1]
        c = m1[1,0]
        d = m1[1,1]

        e = m2[0,0]
        f = m2[0,1]
        g = m2[1,0]
        h = m2[1,1]

        s1 = (b-d)*(g+h)
        s2 = (a+d)*(e+h)
        s3 = (a-c)*(e+f)
        s4 = h*(a+b)
        s5 = (f-h)*a
        s6 = d*(g-e)
        s7 = (c+d)*e

        n11 = s1 + s2 - s4 + s6
        n12 = s4 + s5
        n21 = s6 + s7
        n22 = s2 - s3 + s5 - s7
        return np.array([[n11,n12],[n21,n22]])

    # else break into block matrices and run recursively
    else:
        a,b,c,d = get_blocks(m1)
        e,f,g,h = get_blocks(m2)

        s1 = strassen((b-d), (g+h))
        np.testing.assert_equal(s1, (b-d)@(g+h))
        s2 = strassen((a+d), (e+h))
        np.testing.assert_equal(s2, (a+d)@(e+h))
        s3 = strassen((a-c), (e+f))
        np.testing.assert_equal(s3, (a-c)@(e+f))
        s4 = strassen((a+b), h)
        np.testing.assert_equal(s4, (a+b)@h)
        s5 = strassen(a, (f-h))
        np.testing.assert_equal(s5, a@(f-h))
        s6 = strassen(d, (g-e))
        np.testing.assert_equal(s6, d@(g-e))
        s7 = strassen((c+d), e)
        np.testing.assert_equal(s7, (c+d)@e)

        n11 = s1 + s2 - s4 + s6
        n12 = s4 + s5
        n21 = s6 + s7
        n22 = s2 - s3 + s5 - s7

        return np.block([[n11,n12],[n21,n22]])

### Tests

## test blocks

def test_blocks_4x4_1():
    blocks = [np.eye(2), np.zeros((2,2)), np.zeros((2,2)), np.eye(2)]
    m = np.eye((4))

    for i,t in enumerate(zip(blocks, get_blocks(m))):
        np.testing.assert_equal(t[0], t[1])

def test_blocks_4x4_2():
    blocks = [np.eye(2), 2*np.eye(2),3*np.eye(2),4*np.eye(2)]
    m = np.block([[blocks[0],blocks[1]],[blocks[2], blocks[3]]])

    for i,t in enumerate(zip(blocks, get_blocks(m))):
        np.testing.assert_equal(t[0], t[1])

def test_blocks_4x4_rand():
    blocks = [np.random.randint(-100, high=100, size=(2,2)) for _ in range(0,4)]
    m = np.block([[blocks[0], blocks[1]], [blocks[2], blocks[3]]])

    for i,t in enumerate(zip(blocks, get_blocks(m))):
        np.testing.assert_equal(t[0], t[1])

## Test 2x2 base case

def test_2x2_zero():
    m1 = np.zeros((2,2), dtype=int)
    m2 = np.zeros((2,2), dtype=int)

    strassen_result = strassen(m1,m2)
    numpy_result = m1@m2

    np.testing.assert_equal(strassen_result, numpy_result)


def test_2x2_rand():
    m1 = np.random.randint(-100, high=100, size=(2,2))
    m2 = np.random.randint(-100, high=100, size=(2,2))

    strassen_result = strassen(m1,m2)
    numpy_result = m1@m2

    np.testing.assert_equal(strassen_result, numpy_result)

## Test 4x4 recursive case

def test_4x4_ones():
    m1 = np.ones((4,4))
    m2 = np.ones((4,4))

    strassen_result = strassen(m1,m2)
    numpy_result = m1@m2

    np.testing.assert_equal(strassen_result, numpy_result)

def test_4x4_increasing_blocks():
    m1 = np.eye((4))
    m2 = np.block([[np.ones((2,2)), 2*np.ones((2,2))],[3*np.ones((2,2)),4*np.ones((2,2))]])

    strassen_result = strassen(m1,m2)
    numpy_result = m1@m2

    np.testing.assert_equal(strassen_result, numpy_result)

    # swap order

    strassen_result = strassen(m2,m1)
    numpy_result = m2@m1

    np.testing.assert_equal(strassen_result, numpy_result)
    
    # block with block

    strassen_result = strassen(m2,m2)
    numpy_result = m2@m2

    np.testing.assert_equal(strassen_result, numpy_result)

def test_4x4_range():
    m1 = np.reshape(range(0,16), (4,4))
    m2 = np.eye(4)

    strassen_result = strassen(m1,m2)
    numpy_result = m1@m2

    np.testing.assert_equal(strassen_result, numpy_result)

    # swap order

    strassen_result = strassen(m2,m1)
    numpy_result = m2@m1

    np.testing.assert_equal(strassen_result, numpy_result)

    # range with range

    strassen_result = strassen(m1,m1)
    numpy_result = m1@m1

    np.testing.assert_equal(strassen_result, numpy_result)

def test_4x4_rand():
    m1 = np.random.randint(-10, high=10, size=(4,4))
    m2 = np.random.randint(-10, high=10, size=(4,4))

    strassen_result = strassen(m1,m2)
    numpy_result = m1@m2

    np.testing.assert_equal(strassen_result, numpy_result)

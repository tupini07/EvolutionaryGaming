import itertools
import math
######
import types
from functools import wraps

import numpy as np
import scipy
import scipy.stats
from scipy.ndimage.filters import median_filter

import cv2

functions_atari = ["add", "lt", "average", "aminus", "mult", "cmult1", "cmult2", "inv1", "inv2", "abs1", "abs2", "sqrt1", "sqrt2", "cpow1", "cpow2", "ypow", "exp1", "exp2",
                   "sin1", "sin2", "sqrtxy", "acos1", "acos2", "asin1", "asin2", "atan1", "atan2", "max2", "min2",
                   "split_before",
                   "split_after",
                   "range_in",
                   "index_y",
                   "index_p",
                   "vectorize",
                   "first",
                   "last",
                   "rotate",
                   "push_back",
                   "push_front",
                   "set_fs",
                   "sum_fs",
                   "transpose",
                   "vecfromdouble",
                   "y_wire",
                   "no_op",
                   "const_fs",
                   "constvectord",
                   "zeros",
                   "ones",
                   "reverse"]

statistical_functions = [
    "stddev", "skew", "kurtosis", "mean", "range", "round_st", "ceil", "floor", "max1", "min1"
]

functions_openCV = [
    "GaussianBlur",
    "Sobel1", "Sobel2",
    "Sobelx1", "Sobelx2",
    "Sobely1", "Sobely2",
    "smoothMedian1", "smoothMedian2",
    "smoothBilateral1", "smoothBilateral2",
    "unsharpen1", "unsharpen2",
    "shiftLeft1", "shiftLeft2",
    "shiftRight1", "shiftRight2",
    "shiftUp1", "shiftUp2",
    "shiftDown1", "shiftDown2",
    "shift1", "shift2",
    "erosion1", "erosion2",
    "dilation1", "dilation2",
    "threshold1", "threshold2",
    "reScale1", "reScale2",
    "gabor1", "gabor2",
    "resizeThenGabor1", "resizeThenGabor2",
    "laplace1", "laplace2",
    "canny1", "canny2",
    "localNormalize1", "localNormalize2",
    "differences1", "differences2",
    "avgDifferences1", "avgDifferences2"
]

functions = functions_atari + statistical_functions + functions_openCV

# np.seterr(all="raise")
np.seterr(all="ignore")


def _is_matrix_matrix(inp1, inp2):
    return isinstance(inp1, np.ndarray) and isinstance(inp2, np.ndarray)


def _is_matrix_scalar(inp1, inp2):
    return isinstance(inp1, np.ndarray) and isinstance(inp2, (int, float))


def _is_scalar_matrix(inp1, inp2):
    return isinstance(inp1, (int, float)) and isinstance(inp2, np.ndarray)


def _make_matrices_same_size(inp1, inp2):
    """
    makes matrices of different sizes to be the same size

    Parameters
    ----------
    inp1 : np.ndarray
    inp2 : np.ndarray

    Returns
    -------
        [inp1, inp2]
        resized to the same size, by padding the smaller dimensions
    """

    if inp1.shape == inp2.shape:
        return inp1, inp2

    # if it is different sizes then just resize the arrays to be of the same size as the smaller one
    smallest_shape = [min(p) for p in zip(inp1.shape, inp2.shape)]

    return [np.resize(x, smallest_shape) for x in [inp1, inp2]]

    # below is a possibility of how to pad to the biggest size. But making it equal in the biggest size is a
    # bit harder, and most importantly may add noise. Although with the smallest method used we lose some information
    # big_shape = [max(p) for p in itertools.zip_longest(inp1.shape, inp2.shape, fillvalue=0)]

    # def pad_to_big_size(arr):
    #     differences = [(0, b_size - m_size) for b_size, m_size in itertools.zip_longest(big_shape, arr.shape, fillvalue=0)]
    #     return np.pad(arr, differences, 'constant')


def add(inp1, inp2, parameter, combination_type=None):
    """
    Add to input values.
    If both inputs are arrays, they're made the same shape and then added together.
    If the inputs are a scalar and a matrix then the scalar is added element wise to 
    the elements of the matrix.

    Parameters
    ----------
    inp1 : float or np.ndarray
        First input value.
    inp2 : float or np.ndarray
        Second input value.
    parameter : float
        This is the parameter function. For this method this parameter is simply ignored
        Not actually used by this function.

    Return
    ------
    added : float or np.ndarray
        Sum of inputs.
    """
    if _is_matrix_matrix(inp1, inp2):
        inp1, inp2 = _make_matrices_same_size(inp1, inp2)

    return np.add(inp1, inp2)


def lt(inp1, inp2, parameter):
    """
    Get lesser of the two input values.
    If one or both input values are arrays,
    their means will be used instead.

    Parameters
    ----------
    inp1 : float or np.ndarray
        First input value.
    inp2 : float or np.ndarray
        Second input value.
    parameter : float
        Not actually used by this function.

    Return
    ------
    lesser : float
        Lesser of the two values.
    """

    if _is_matrix_matrix(inp1, inp2):
        m1, m2 = _make_matrices_same_size(inp1, inp2)
        return m1 < m2

    else:  # either matrix scalar, or viceversa, or scalar scalar
        return inp1 < inp2


def GaussianBlur(inp1, inp2, parameter):
    """
    Blurs inp1 using a Gaussian filter.

    Parameters
    ----------
    inp1 : float or np.ndarray
        First input value.
    inp2 : float or np.ndarray
        Second input value.
    parameter : float
        Used to determne kernel size.

    Return
    ------
    blurred : np.ndarray
        Blurred version of inp1.
    """

    if not isinstance(inp1, np.ndarray) or len(inp1.shape) != 2:
        return inp1

    if parameter < 0:
        parameter *= -1

    ksizex = int(parameter * inp1.shape[0])
    if ksizex % 2 == 0:
        ksizex += 1

    ksizey = int(parameter * inp1.shape[1])
    if ksizey % 2 == 0:
        ksizey += 1

    inp1 = np.float32(inp1)
    blurred = cv2.GaussianBlur(inp1, (ksizex, ksizey), 0, sigmaY=0)

    return blurred


def average(inp1, inp2, parameter):
    """
    Calculate (inp1 + inp2)/2.
    This is the add function from Wilson et.al.

    Parameters
    ----------
    inp1 : float or np.ndarray
        First input value.
    inp2 : float or np.ndarray
        Second input value.
    parameter : float
        Not actually used by this function.

    Return
    ------
    avg : float or np.ndarray
        (inp1 + inp2)/2
    """

    inp_sum = add(inp1, inp2, parameter)

    avg = inp_sum / 2

    return avg


def aminus(inp1, inp2, parameter):
    """
    Calculates |inp1 - inp2|/2.
    Function from Wilson et.al.

    Parameters
    ----------
    inp1 : float or np.ndarray
        First input value.
    inp2 : float or np.ndarray
        Second input value.
    parameter : float
        Not actually used by this function.

    Return
    ------
    avg : float or np.ndarray
        |inp1 - inp2|/2
    """

    if _is_matrix_matrix(inp1, inp2):
        inp1, inp2 = _make_matrices_same_size(inp1, inp2)

    diff = np.abs(np.subtract(inp1, inp2))

    return np.divide(diff, 2)


def mult(inp1, inp2, parameter):
    """
    Multiply inp1 and inp2.
    Function from Wilson et.al.

    Parameters
    ----------
    inp1 : float or np.ndarray
        First input value.
    inp2 : float or np.ndarray
        Second input value.
    parameter : float
        Not actually used by this function.

    Return
    ------
    prod : float or np.ndarray
        inp1 * inp2
    """

    if _is_matrix_matrix(inp1, inp2):
        inp1, inp2 = _make_matrices_same_size(inp1, inp2)

    return np.multiply(inp1, inp2)


def cmult1(inp1, inp2, parameter):
    """
    Multiply inp1 by parameter.
    Function from Wilson et.al.

    Parameters
    ----------
    inp1 : float or np.ndarray
        First input value.
    inp2 : float or np.ndarray
        Second input value. Not actually used by this function.
    parameter : float
        Evolved parameter by which to multiply inp1.

    Return
    ------
    prod : float or np.ndarray
        inp1 * parameter
    """

    return np.multiply(inp1, parameter)


def cmult2(inp1, inp2, parameter):
    """
    Multiply inp2 by parameter.
    Function from Wilson et.al.

    Parameters
    ----------
    inp1 : float or np.ndarray
        First input value. Not actually used by this function.
    inp2 : float or np.ndarray
        Second input value.
    parameter : float
        Evolved parameter by which to multiply inp2.

    Return
    ------
    prod : float or np.ndarray
        inp2 * parameter
    """

    return np.multiply(inp2, parameter)


def inv1(inp1, inp2, parameter):
    """
    Calculate inverse of inp1.
    Marices that aren't square matrices are padded with zeros before inversion.
    Matrices that can't be inverted are passed to the output.
    1/inp1 for scalar inputs.
    Function from Wilson et.al.

    Parameters
    ----------
    inp1 : float or np.ndarray
        First input value.
    inp2 : float or np.ndarray
        Second input value. Not actually used by this function.
    parameter : float
        Not actually used by this function.

    Return
    ------
    inv : float
        Inverted inp1.
    """

    inp1 = np.array(inp1)
    return np.nan_to_num(np.divide(1, inp1))


def inv2(inp1, inp2, parameter):
    """
    Calculate inverse of inp2.
    Marices that aren't square matrices are padded with zeros before inversion.
    Matrices that can't be inverted are passed to the output.
    1/inp2 for scalar inputs.
    Function from Wilson et.al.

    Parameters
    ----------
    inp1 : float or np.ndarray
        First input value. Not actually used by this function.
    inp2 : float or np.ndarray
        Second input value.
    parameter : float
        Not actually used by this function.

    Return
    ------
    inv : float
        Inverted inp2.
    """

    inp2 = np.array(inp2)
    return np.nan_to_num(np.divide(1, inp2))


def abs1(inp1, inp2, parameter):
    """
    Return absolute value of inp1.
    If inp1 is an array, abs is applied element-wise.
    Function from Wilson et.al.

    Parameters
    ----------
    inp1 : float or np.ndarray
        First input value.
    inp2 : float or np.ndarray
        Second input value. Not actually used by this function.
    parameter : float
        Not actually used by this function.

    Return
    ------
    abs1 : float or np.ndarray
        Absolute value of inp1.
    """

    abs1 = np.abs(inp1)

    return abs1


def abs2(inp1, inp2, parameter):
    """
    Return absolute value of inp2.
    If inp2 is an array, abs is applied element-wise.
    Function from Wilson et.al.

    Parameters
    ----------
    inp1 : float or np.ndarray
        First input value. Not actually used by this function.
    inp2 : float or np.ndarray
        Second input value.
    parameter : float
        Not actually used by this function.

    Return
    ------
    abs2 : float or np.ndarray
        Absolute value of inp1.
    """

    abs2 = abs(inp2)

    return abs2


def sqrt1(inp1, inp2, parameter):
    """
    Return square root of inp1.
    If inp1 is an array, the operation is applied element-wise.
    Negative elements are replaced with their absolute values first.
    Function from Wilson et.al.

    Parameters
    ----------
    inp1 : float or np.ndarray
        First input value.
    inp2 : float or np.ndarray
        Second input value. Not actually used by this function.
    parameter : float
        Not actually used by this function.

    Return
    ------
    sqrt1 : float or np.ndarray
        Square root of inp1.
    """

    abs1 = np.abs(inp1)

    sqrt1 = np.sqrt(abs1)

    return sqrt1


def sqrt2(inp1, inp2, parameter):
    """
    Return square root of inp2.
    If inp2 is an array, the operation is applied element-wise.
    Negative elements are replaced with their absolute values first.
    Function from Wilson et.al.

    Parameters
    ----------
    inp1 : float or np.ndarray
        First input value.
    inp2 : float or np.ndarray
        Second input value. Not actually used by this function.
    parameter : float
        Not actually used by this function.

    Return
    ------
    sqrt2 : float or np.ndarray
        Square root of inp2.
    """

    abs2 = np.abs(inp2)

    sqrt2 = np.sqrt(abs2)

    return sqrt2


def cpow1(inp1, inp2, parameter):
    """
    Calculate |inp1| to the power of parameter.
    If inp1 is a matrix, it is padded to be square.
    Parameter is converted to its absolute value before the operation.
    Function from Wilson et.al.

    Parameters
    ----------
    inp1 : float or np.ndarray
        First input value.
    inp2 : float or np.ndarray
        Second input value. Not actually used by this function.
    parameter : float
        Not actually used by this function.

    Return
    ------
    cpow1 : float or np.ndarray
        |inp1|^parameter.
    """

    return np.power(np.absolute(inp1), parameter + 1)


def cpow2(inp1, inp2, parameter):
    """
    Calculate |inp2| to the power of parameter.
    If inp2 is a matrix, it is padded to be square.
    Parameter is converted to its absolute value before the operation.
    Function from Wilson et.al.

    Parameters
    ----------
    inp1 : float or np.ndarray
        First input value. Not actually used by this function.
    inp2 : float or np.ndarray
        Second input value.
    parameter : float
        Not actually used by this function.

    Return
    ------
    cpow2 : float or np.ndarray
        |inp2|^parameter.
    """

    return np.power(np.absolute(inp2), parameter + 1)


def ypow(inp1, inp2, parameter):
    """
    Calculate |inp1|^|inp2|.
    If inp2 is a matrix, its mean is taken beforehand.
    If inp1 is a matrix, it is zero-padded to square form.
    Function from Wilson et.al.

    Parameters:
    -----------
    inp1 : float or np.ndarray
        Base of exponentiation.
    inp2 : float or np.ndarray
        Exponent of exponentiation.
    parameter : float
        Not actually used by this function.

    Return
    ------
    ypow : float or np.ndarray
        |inp1|^|inp2|
    """

    if isinstance(inp2, np.ndarray):
        inp2 = np.mean(inp2)

    return np.power(inp1, inp2)


def exp1(inp1, inp2, parameter):
    """
    Calculate (e^inp1 - 1) / (e - 1).
    If inp1 is an array, the operation is applied element-wise.
    EXPX Function from Wilson et.al.

    Parameters:
    -----------
    inp1 : float or np.ndarray
        Base of exponentiation.
    inp2 : float or np.ndarray
        Exponent of exponentiation. Not actually used by this function.
    parameter : float
        Not actually used by this function.

    Return
    ------
    exp1 : float
        (e^inp1 - 1) / (e - 1)
    """

    exp1 = np.expm1(inp1)
    exp1 = exp1 / (np.exp(1) - 1)

    return exp1


def exp2(inp1, inp2, parameter):
    """
    Calculate (e^inp2 - 1) / (e - 1).
    If inp2 is an array, the operation is applied element-wise.
    EXPX Function from Wilson et.al.

    Parameters:
    -----------
    inp1 : float or np.ndarray
        Base of exponentiation. Not actually used by this function.
    inp2 : float or np.ndarray
        Exponent of exponentiation.
    parameter : float
        Not actually used by this function.

    Return
    ------
    exp2 : float
        (e^inp2 - 1) / (e - 1)
    """

    exp2 = np.expm1(inp2)
    exp2 = np.divide(exp2, (np.exp(1) - 1))

    return exp2


def sin1(inp1, inp2, parameter):
    """
    Get sin of first input value.
    If it is an array, sine will be applied element-wise.
    Function from Wilson et.al.

    Parameters
    ----------
    inp1 : float or np.ndarray
        First input value.
    inp2 : float or np.ndarray
        Second input value. Not actually used by this function.
    parameter : float
        Not actually used by this function.

    Return
    ------
    sin : float
        Sine of first input value.
    """

    sin = np.sin(inp1)

    return sin


def sin2(inp1, inp2, parameter):
    """
    Get sin of second input value.
    If it is an array, sine will be applied element-wise.
    Function from Wilson et.al.

    Parameters
    ----------
    inp1 : float or np.ndarray
        First input value. Not actually used by this function.
    inp2 : float or np.ndarray
        Second input value.
    parameter : float
        Not actually used by this function.

    Return
    ------
    sin : float
        Sine of second input value.
    """

    sin = np.sin(inp2)

    return sin


def sqrtxy(inp1, inp2, parameter):
    """
    Calculate sqrt(inp1^2 + inp2^2) / sqrt(2).
    In case of an array the operation will be applied element-wise.
    Function from Wilson et.al.

    Parameters
    ----------
    inp1 : float or np.ndarray
        First input value.
    inp2 : float or np.ndarray
        Second input value.
    parameter : float
        Not actually used by this function.

    Return
    ------
    sqrtxy : float or np.ndarray
        sqrt(inp1^2 + inp2^2) / sqrt(2)
    """

    if _is_matrix_matrix(inp1, inp2):
        inp1, inp2 = _make_matrices_same_size(inp1, inp2)

    square1 = np.power(inp1, 2)
    square2 = np.power(inp2, 2)

    sum_squares = square1 + square2
    root = np.sqrt(sum_squares)

    sqrtxy = root / np.sqrt(2)

    return sqrtxy


def acos1(inp1, inp2, parameter):
    """
    Calculate arccos(inp1) / pi.
    If inp1 is an array, the operation is applied element-wise.
    Function from Wilson et.al.

    Parameters
    ----------
    inp1 : float or np.ndarray
        First input value.
    inp2 : float or np.ndarray
        Second input value. Not actually used by this function.
    parameter : float
        Not actually used by this function.

    Return
    ------
    acos1 : float or np.ndarray
        arccos(inp1) / pi
    """

    inp1 = np.clip(inp1, -1, 1)
    acos1 = np.arccos(inp1) / np.pi

    return acos1


def acos2(inp1, inp2, parameter):
    """
    Calculate arccos(inp2) / pi.
    If inp2 is an array, the operation is applied element-wise.
    Function from Wilson et.al.

    Parameters
    ----------
    inp1 : float or np.ndarray
        First input value. Not actually used by this function.
    inp2 : float or np.ndarray
        Second input value.
    parameter : float
        Not actually used by this function.

    Return
    ------
    acos2 : float or np.ndarray
        arccos(inp2) / pi
    """

    inp2 = np.clip(inp2, -1, 1)
    acos2 = np.arccos(inp2) / np.pi

    return acos2


def asin1(inp1, inp2, parameter):
    """
    Calculate 2 * arcsin(inp1) / pi.
    If inp1 is an array, the operation is applied element-wise.
    Function from Wilson et.al.

    Parameters
    ----------
    inp1 : float or np.ndarray
        First input value.
    inp2 : float or np.ndarray
        Second input value. Not actually used by this function.
    parameter : float
        Not actually used by this function.

    Return
    ------
    asin1 : float or np.ndarray
        2 * arcsin(inp1) / pi
    """

    inp1 = np.clip(inp1, -1, 1)
    asin1 = 2 * np.arcsin(inp1) / np.pi

    return asin1


def asin2(inp1, inp2, parameter):
    """
    Calculate 2 * arcsin(inp2) / pi.
    If inp2 is an array, the operation is applied element-wise.
    Function from Wilson et.al.

    Parameters
    ----------
    inp1 : float or np.ndarray
        First input value. Not actually used by this function.
    inp2 : float or np.ndarray
        Second input value.
    parameter : float
        Not actually used by this function.

    Return
    ------
    asin2 : float or np.ndarray
        2 * arcsin(inp2) / pi
    """

    inp2 = np.clip(inp2, -1, 1)
    asin2 = 2 * np.arcsin(inp2) / np.pi

    return asin2


def atan1(inp1, inp2, parameter):
    """
    Calculate 4 * arctan(inp1) / pi.
    If inp1 is an array, the operation is applied element-wise.
    Function from Wilson et.al.

    Parameters
    ----------
    inp1 : float or np.ndarray
        First input value.
    inp2 : float or np.ndarray
        Second input value. Not actually used by this function.
    parameter : float
        Not actually used by this function.

    Return
    ------
    atan1 : float or np.ndarray
        4 * arctan(inp1) / pi
    """

    atan1 = 4 * np.arctan(inp1) / np.pi

    return atan1


def atan2(inp1, inp2, parameter):
    """
    Calculate 4 * arctan(inp2) / pi.
    If inp2 is an array, the operation is applied element-wise.
    Function from Wilson et.al.

    Parameters
    ----------
    inp1 : float or np.ndarray
        First input value. Not actually used by this function.
    inp2 : float or np.ndarray
        Second input value.
    parameter : float
        Not actually used by this function.

    Return
    ------
    atan2 : float or np.ndarray
        4 * arctan(inp2) / pi
    """

    atan2 = 4 * np.arctan(inp2) / np.pi

    return atan2


def stddev(inp1, inp2, parameter):
    """
    calculate the standard deviation of `inp1`

    Parameters
    ----------
    inp1 : float or np.ndarray
        First input value. Not actually used by this function.
    inp2 : float or np.ndarray
        Second input value.
    parameter : float
        Not actually used by this function.

    Return
    ------
        stddev: float
            The standard deviation of the matrix inp1 if it is a matrix, 
            else it just returns the number (int or float)    
    """

    if isinstance(inp1, np.ndarray):
        return inp1.std()

    else:
        return inp1


def skew(inp1, inp2, parameter):
    if isinstance(inp1, np.ndarray):
        return scipy.stats.skew(inp1)

    else:
        return inp1


def kurtosis(inp1, inp2, parameter):
    if isinstance(inp1, np.ndarray):
        return scipy.stats.kurtosis(inp1)

    else:
        return inp1


def mean(inp1, inp2, parameter):
    if isinstance(inp1, np.ndarray):
        return inp1.mean()

    else:
        return inp1


def range(inp1, inp2, parameter):
    if isinstance(inp1, np.ndarray):
        return inp1.max() - inp1.min() - 1

    else:
        return inp1


def round_st(inp1, inp2, parameter):
    return np.round(inp1)


def ceil(inp1, inp2, parameter):
    return np.ceil(inp1)


def floor(inp1, inp2, parameter):
    return np.floor(inp1)


def max1(inp1, inp2, parameter):
    if isinstance(inp1, np.ndarray) and len(inp1) > 0:
        return inp1.max()

    else:
        return inp1


def min1(inp1, inp2, parameter):
    if isinstance(inp1, np.ndarray) and len(inp1) > 0:
        return inp1.min()

    else:
        return inp1


def max2(inp1, inp2, parameter):
    if _is_matrix_matrix(inp1, inp2):
        inp1, inp2 = _make_matrices_same_size(inp1, inp2)

    return np.maximum(inp1, inp2)


def min2(inp1, inp2, parameter):
    if _is_matrix_matrix(inp1, inp2):
        inp1, inp2 = _make_matrices_same_size(inp1, inp2)

    return np.minimum(inp1, inp2)


def split_before(inp1, inp2, parameter):
    if isinstance(inp1, np.ndarray):
        idx_p = (parameter + 1) / 2
        split_point = int(np.round(len(inp1) * idx_p))

        return inp1[:split_point]

    else:
        return inp1


def split_after(inp1, inp2, parameter):
    if isinstance(inp1, np.ndarray):
        idx_p = (parameter + 1) / 2
        split_point = int(np.round(len(inp1) * idx_p))

        return inp1[split_point:]

    else:
        return inp1


def range_in(inp1, inp2, parameter):
    if isinstance(inp1, np.ndarray):
        inp1 = np.nan_to_num(inp1)
        inp2 = np.nan_to_num(inp2)
        parameter = np.nan_to_num(parameter)

        if isinstance(inp2, np.ndarray):
            inp2 = inp2.mean()

        inp2 = np.clip(inp2, -1, 1)

        right_p = (np.max([inp2, parameter]) + 1) / 2
        left_p = (np.min([inp2, parameter]) + 1) / 2

        split_right = int(np.round(len(inp1) * right_p))
        split_left = int(np.round(len(inp1) * left_p))

        if split_left == split_right:
            split_right += 1

        return inp1[split_left:split_right]

    else:
        return inp1


def index_y(inp1, inp2, parameter):
    if isinstance(inp1, np.ndarray) and len(inp1) > 0:
        if isinstance(inp2, np.ndarray):
            inp2 = inp2.mean()

        inp2 = np.clip(inp2, -1, 1)

        i_p = (inp2 + 1) / 2
        split_point = int(np.round(len(inp1) * i_p)) - 1

        return inp1[split_point]

    else:
        return inp1


def index_p(inp1, inp2, parameter):
    if isinstance(inp1, np.ndarray) and len(inp1) > 0:
        idx_p = (parameter + 1) / 2
        split_point = int(np.round(len(inp1) * idx_p)) - 1

        return inp1[split_point]

    else:
        return inp1


def vectorize(inp1, inp2, parameter):
    if isinstance(inp1, np.ndarray):
        return inp1.flatten()

    else:
        return inp1


def first(inp1, inp2, parameter):
    if isinstance(inp1, np.ndarray) and len(inp1) > 0:
        return inp1[0]

    else:
        return inp1


def last(inp1, inp2, parameter):
    if isinstance(inp1, np.ndarray) and len(inp1) > 0:
        return inp1[-1]

    else:
        return inp1

# TODO: differences -> computational derivative of 1D vector of inp1


def rotate(inp1, inp2, parameter):
    if isinstance(inp1, np.ndarray):
        num_to_rotate = round(len(inp1) * ((parameter + 1) / 2))
        return np.roll(inp1, num_to_rotate, axis=0)

    else:
        return inp1


def reverse(inp1, inp2, parameter):
    if isinstance(inp1, np.ndarray):
        return np.flip(inp1)  # flip along all axis

    else:
        return inp1


def push_back(inp1, inp2, parameter):
    return np.append(inp1, inp2)


def push_front(inp1, inp2, parameter):
    return np.append(inp2, inp1)


def set_fs(inp1, inp2, parameter):
    if _is_matrix_matrix(inp1, inp2):
        return np.array([inp1[0]] * len(inp2))

    elif _is_matrix_scalar(inp1, inp2):
        return np.array([inp2] * len(inp1))

    elif _is_scalar_matrix(inp1, inp2):
        return np.array([inp1] * len(inp2))

    else:
        return np.array([inp1])


def sum_fs(inp1, inp2, parameter):
    if isinstance(inp1, np.ndarray):
        return np.sum(inp1)

    else:
        return inp1


def transpose(inp1, inp2, parameter):
    if isinstance(inp1, np.ndarray):
        return np.transpose(inp1)

    else:
        return inp1


def vecfromdouble(inp1, inp2, parameter):
    if isinstance(inp1, (int, float)):
        return np.array([inp1])

    else:
        return inp1


def y_wire(inp1, inp2, parameter):
    return inp2


def no_op(inp1, inp2, parameter):
    return inp1


def const_fs(inp1, inp2, parameter):
    return parameter


def constvectord(inp1, inp2, parameter):
    if isinstance(inp1, np.ndarray):
        return np.zeros(inp1.shape) + parameter

    else:
        return np.array([parameter])


def zeros(inp1, inp2, parameter):
    if isinstance(inp1, np.ndarray):
        return np.zeros(inp1.shape)

    else:
        return np.array([0])


def ones(inp1, inp2, parameter):
    if isinstance(inp1, np.ndarray):
        return np.ones(inp1.shape)

    else:
        return np.array([1])


######


def Sobel1(inp1, inp2, parameter):
    """
    Calculate first derivative of inp1 with an extended Sobel operator.
    See opencv documentation for further explanation.

    Parameters
    ----------
    inp1 : float or np.ndarray
        First input value.
    inp2 : float or np.ndarray
        Second input value. Not actually used by this function.
    parameter : float
        Not actually used by this function.

    Return
    ------
    sobel : np.ndarray
        Image after application of Sobel operator.
    """

    if not isinstance(inp1, np.ndarray):
        return inp1

    mat_array_1 = cv2.UMat(np.float32(inp1))

    sobel = cv2.Sobel(mat_array_1, -1, 1, 1)

    sobel = sobel.get()

    return sobel


def Sobel2(inp1, inp2, parameter):
    """
    Calculate first derivative of inp2 with an extended Sobel operator.
    See opencv documentation for further explanation.

    Parameters
    ----------
    inp1 : float or np.ndarray
        First input value. Not actually used by this function.
    inp2 : float or np.ndarray
        Second input value.
    parameter : float
        Not actually used by this function.

    Return
    ------
    sobel : np.ndarray
        Image after application of Sobel operator.
    """

    if not isinstance(inp2, np.ndarray):
        return inp2

    mat_array_2 = cv2.UMat(np.float32(inp2))

    sobel = cv2.Sobel(mat_array_2, -1, 1, 1)

    sobel = sobel.get()

    return sobel


def Sobelx1(inp1, inp2, parameter):
    """
    Calculate first derivative of inp1 
    with an extended Sobel operator over the x dimension.
    See opencv documentation for further explanation.

    Parameters
    ----------
    inp1 : float or np.ndarray
        First input value.
    inp2 : float or np.ndarray
        Second input value. Not actually used by this function.
    parameter : float
        Not actually used by this function.

    Return
    ------
    sobel : np.ndarray
        Image after application of Sobel operator.
    """

    if not isinstance(inp1, np.ndarray):
        return inp1

    mat_array_1 = cv2.UMat(np.float32(inp1))

    sobel = cv2.Sobel(mat_array_1, -1, 1, 0)

    sobel = sobel.get()

    return sobel


def Sobelx2(inp1, inp2, parameter):
    """
    Calculate first derivative of inp2 
    with an extended Sobel operator over the x dimension.
    See opencv documentation for further explanation.

    Parameters
    ----------
    inp1 : float or np.ndarray
        First input value. Not actually used by this function.
    inp2 : float or np.ndarray
        Second input value.
    parameter : float
        Not actually used by this function.

    Return
    ------
    sobel : np.ndarray
        Image after application of Sobel operator.
    """

    if not isinstance(inp2, np.ndarray):
        return inp2

    mat_array_2 = cv2.UMat(np.float32(inp2))

    sobel = cv2.Sobel(mat_array_2, -1, 1, 0)

    sobel = sobel.get()

    return sobel


def Sobely1(inp1, inp2, parameter):
    """
    Calculate first derivative of inp1 
    with an extended Sobel operator over the y dimension.
    See opencv documentation for further explanation.

    Parameters
    ----------
    inp1 : float or np.ndarray
        First input value.
    inp2 : float or np.ndarray
        Second input value. Not actually used by this function.
    parameter : float
        Not actually used by this function.

    Return
    ------
    sobel : np.ndarray
        Image after application of Sobel operator.
    """

    if not isinstance(inp1, np.ndarray):
        return inp1

    mat_array_1 = cv2.UMat(np.float32(inp1))

    sobel = cv2.Sobel(mat_array_1, -1, 0, 1)

    sobel = sobel.get()

    return sobel


def Sobely2(inp1, inp2, parameter):
    """
    Calculate first derivative of inp2 
    with an extended Sobel operator over the y dimension.
    See opencv documentation for further explanation.

    Parameters
    ----------
    inp1 : float or np.ndarray
        First input value. Not actually used by this function.
    inp2 : float or np.ndarray
        Second input value.
    parameter : float
        Not actually used by this function.

    Return
    ------
    sobel : np.ndarray
        Image after application of Sobel operator.
    """

    if not isinstance(inp2, np.ndarray):
        return inp2

    mat_array_2 = cv2.UMat(np.float32(inp2))

    sobel = cv2.Sobel(mat_array_2, -1, 0, 1)

    sobel = sobel.get()

    return sobel


def threshold1(inp1, inp2, parameter):
    """
    Go over inp1, replace each pixel with 1 if it has a higher value
    than parameter, with 0 otherwise.

    Parameters
    ----------
    inp1 : float or np.ndarray
        First input value.
    inp2 : float or np.ndarray
        Second input value. Not actually used by this function.
    parameter : float
        Value to compare the pixels to.

    Return
    ------
    black_and_white : float or np.ndarray
        inp1 after the pixels were replaced.
    """

    if not isinstance(inp1, np.ndarray):
        return inp1

    with np.nditer(inp1, op_flags=['readwrite']) as it:
        for x in it:
            if x > parameter:
                x[...] = -1
            else:
                x[...] = 1

    black_and_white = inp1

    return inp1


def threshold2(inp1, inp2, parameter):
    """
    Go over inp2, replace each pixel with 1 if it has a higher value
    than parameter, with 0 otherwise.

    Parameters
    ----------
    inp1 : float or np.ndarray
        First input value. Not actually used by this function.
    inp2 : float or np.ndarray
        Second input value.
    parameter : float
        Value to compare the pixels to.

    Return
    ------
    black_and_white : float or np.ndarray
        inp2 after the pixels were replaced.
    """

    if not isinstance(inp2, np.ndarray):
        return inp2

    with np.nditer(inp2, op_flags=['readwrite']) as it:
        for x in it:
            if x > parameter:
                x[...] = -1
            else:
                x[...] = 1

    black_and_white = inp2

    return inp2


def smoothMedian1(inp1, inp2, parameter):
    """
    Blur inp1 using a kernel that replaces the pixel values
    with their median.

    Parameters
    ----------
    inp1 : float or np.ndarray
        First input value.
    inp2 : float or np.ndarray
        Second input value. Not actually used by this function.
    parameter : float
        Not actually used by this function.

    Return
    ------
    smoothed : float or np.ndarray
        Blurred image.
    """

    if not isinstance(inp1, np.ndarray):
        return inp1

    # if parameter < 0:
    #    parameter *= -1
    #shape = min(inp1.shape[0], inp1.shape[1])
    #ksize = int(parameter * shape)
    # if ksize % 2 == 0:
    #    ksize += 1

    mat_array_1 = cv2.UMat(np.float32(inp1))
    smoothed = cv2.medianBlur(mat_array_1, ksize=3)
    smoothed = smoothed.get()

    return smoothed


def smoothMedian2(inp1, inp2, parameter):
    """
    Blur inp2 using a kernel that replaces the pixel values
    with their median.

    Parameters
    ----------
    inp1 : float or np.ndarray
        First input value. Not actually used by this function.
    inp2 : float or np.ndarray
        Second input value.
    parameter : float
        Not actually used by this function

    Return
    ------
    smoothed : float or np.ndarray
        Blurred image.
    """

    if not isinstance(inp2, np.ndarray):
        return inp2

    # if parameter < 0:
    #    parameter *= -1
    #shape = min(inp2.shape[0], inp2.shape[1])
    #ksize = int(parameter * shape)
    # if ksize % 2 == 0:
    #    ksize += 1

    mat_array_2 = cv2.UMat(np.float32(inp2))
    smoothed = cv2.medianBlur(mat_array_2, ksize=3)
    smoothed = smoothed.get()

    return smoothed


def smoothBilateral1(inp1, inp2, parameter):
    """
    Blur inp1 using a bilateral filter.

    Parameters
    ----------
    inp1 : float or np.ndarray
        First input value.
    inp2 : float or np.ndarray
        Second input value. Not actually used by this function.
    parameter : float
        Used to determine sigma values.

    Return
    ------
    smoothed : float or np.ndarray
        Blurred image.
    """

    if not isinstance(inp1, np.ndarray):
        return inp1

    if parameter < 0:
        parameter *= -1

    sigma = int(parameter * 175)
    # Note: 175 chosen because the opencv documentation call values > 150 large
    # https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?highlight=bilateralfilter#bilateralfilter

    inp1 = np.float32(inp1)

    smoothed = cv2.bilateralFilter(
        inp1, d=5, sigmaColor=sigma, sigmaSpace=sigma)

    smoothed = np.uint8(smoothed)

    return smoothed


def smoothBilateral2(inp1, inp2, parameter):
    """
    Blur inp2 using a bilateral filter.

    Parameters
    ----------
    inp1 : float or np.ndarray
        First input value. Not actually used by this function.
    inp2 : float or np.ndarray
        Second input value.
    parameter : float
        Used to determine sigma values.

    Return
    ------
    smoothed : float or np.ndarray
        Blurred image.
    """

    if not isinstance(inp2, np.ndarray):
        return inp2

    if parameter < 0:
        parameter *= -1

    sigma = int(parameter * 175)
    # Note: 175 chosen because the opencv documentation call values > 150 large
    # https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?highlight=bilateralfilter#bilateralfilter

    inp2 = np.float32(inp2)

    smoothed = cv2.bilateralFilter(
        inp2, d=5, sigmaColor=sigma, sigmaSpace=sigma)

    smoothed = np.uint8(smoothed)

    return smoothed


def smoothBlur1(inp1, inp2, parameter):
    """
    Blur inp1 using a normalized box filter.

    Parameters
    ----------
    inp1 : float or np.ndarray
        First input value.
    inp2 : float or np.ndarray
        Second input value. Not actually used by this function.
    parameter : float
        Used to determine kernel size.

    Return
    ------
    smoothed : float or np.ndarray
        Blurred image.
    """

    if not isinstance(inp1, np.ndarray):
        return inp1

    if parameter < 0:
        parameter *= -1

    ksizex = int(parameter * inp1.shape[0])
    if ksizex % 2 == 0:
        ksizex += 1

    ksizey = int(parameter * inp1.shape[1])
    if ksizey % 2 == 0:
        ksizey += 1

    smoothed = cv2.blur(inp1, (ksizex, ksizey))

    return smoothed


def smoothBlur2(inp1, inp2, parameter):
    """
    Blur inp2 using a normalized box filter.

    Parameters
    ----------
    inp1 : float or np.ndarray
        First input value. Not actually used by this function.
    inp2 : float or np.ndarray
        Second input value.
    parameter : float
        Used to determine kernel size.

    Return
    ------
    smoothed : float or np.ndarray
        Blurred image.
    """

    if not isinstance(inp2, np.ndarray):
        return inp2

    if parameter < 0:
        parameter *= -1

    ksizex = int(parameter * inp2.shape[0])
    if ksizex % 2 == 0:
        ksizex += 1

    ksizey = int(parameter * inp2.shape[1])
    if ksizey % 2 == 0:
        ksizey += 1

    smoothed = cv2.blur(inp2, (ksizex, ksizey))

    return smoothed


def unsharpen1(inp1, inp2, parameter):
    """
    Apply the unsharpen operation to inp1.

    Parameters
    ----------
    inp1 : float or np.ndarray
        First input value.
    inp2 : float or np.ndarray
        Second input value. Not actually used by this function.
    parameter : float
        Weight of substraction.

    Return
    ------
    sharp : float or np.ndarray
        Unsharpened image.
    """

    if not isinstance(inp1, np.ndarray):
        return inp1

    inp1 = np.float32(inp1)
    laplacian = cv2.Laplacian(inp1, cv2.CV_32F)
    # median filter to counteract noise of laplacian
    gray = median_filter(inp1, 1)

    sharp = gray - parameter * laplacian

    return sharp


def unsharpen2(inp1, inp2, parameter):
    """
    Apply the unsharpen operation to inp2.

    Parameters
    ----------
    inp1 : float or np.ndarray
        First input value. Not actually used by this function.
    inp2 : float or np.ndarray
        Second input value.
    parameter : float
        Weight of substraction.

    Return
    ------
    sharp : float or np.ndarray
        Unsharpened image.
    """

    if not isinstance(inp2, np.ndarray):
        return inp2

    inp2 = np.float32(inp2)
    laplacian = cv2.Laplacian(inp2, cv2.CV_32F)
    # median filter to counteract noise of laplacian
    gray = median_filter(inp2, 1)

    sharp = gray - parameter * laplacian

    return sharp


def shiftLeft1(inp1, inp2, parameter):
    """
    Perform a circular left shift on inp1.

    Parameters
    ----------
    inp1 : float or np.ndarray
        First input value.
    inp2 : float or np.ndarray
        Second input value. Not actually used by this function.
    parameter : float
        Determines, how far the image is shifted.

    Return
    ------
    shifted : float or np.ndarray
        Shifted image.
    """

    if not isinstance(inp1, np.ndarray):
        return inp1
    if len(inp1.shape) == 1:
        return inp1

    if parameter < 0:
        parameter *= -1

    dist = int(parameter * inp1.shape[1])

    left = inp1[:, :dist]
    right = inp1[:, dist:]

    shifted = np.append(right, left, axis=1)

    return shifted


def shiftLeft2(inp1, inp2, parameter):
    """
    Perform a circular left shift on inp2.

    Parameters
    ----------
    inp1 : float or np.ndarray
        First input value. Not actually used by this function.
    inp2 : float or np.ndarray
        Second input value.
    parameter : float
        Determines, how far the image is shifted.

    Return
    ------
    shifted : float or np.ndarray
        Shifted image.
    """

    if not isinstance(inp2, np.ndarray):
        return inp2

    if len(inp2.shape) == 1:
        return inp2

    if parameter < 0:
        parameter *= -1

    dist = int(parameter * inp2.shape[1])

    left = inp2[:, :dist]
    right = inp2[:, dist:]

    shifted = np.append(right, left, axis=1)

    return shifted


def shiftRight1(inp1, inp2, parameter):
    """
    Perform a circular right shift on inp1.

    Parameters
    ----------
    inp1 : float or np.ndarray
        First input value.
    inp2 : float or np.ndarray
        Second input value. Not actually used by this function.
    parameter : float
        Determines, how far the image is shifted.

    Return
    ------
    shifted : float or np.ndarray
        Shifted image.
    """

    if not isinstance(inp1, np.ndarray):
        return inp1
    if len(inp1.shape) == 1:
        return inp1

    if parameter < 0:
        parameter *= -1

    dist = inp1.shape[1] - int(parameter * inp1.shape[1])

    left = inp1[:, :dist]
    right = inp1[:, dist:]

    shifted = np.append(right, left, axis=1)

    return shifted


def shiftRight2(inp1, inp2, parameter):
    """
    Perform a circular left shift on inp2.

    Parameters
    ----------
    inp1 : float or np.ndarray
        First input value. Not actually used by this function.
    inp2 : float or np.ndarray
        Second input value.
    parameter : float
        Determines, how far the image is shifted.

    Return
    ------
    shifted : float or np.ndarray
        Shifted image.
    """

    if not isinstance(inp2, np.ndarray):
        return inp2
    if len(inp2.shape) == 1:
        return inp2

    if parameter < 0:
        parameter *= -1

    dist = inp2.shape[1] - int(parameter * inp2.shape[1])

    left = inp2[:, :dist]
    right = inp2[:, dist:]

    shifted = np.append(right, left, axis=1)

    return shifted


def shiftUp1(inp1, inp2, parameter):
    """
    Perform a circular upward shift on inp1.

    Parameters
    ----------
    inp1 : float or np.ndarray
        First input value.
    inp2 : float or np.ndarray
        Second input value. Not actually used by this function.
    parameter : float
        Determines, how far the image is shifted.

    Return
    ------
    shifted : float or np.ndarray
        Shifted image.
    """

    if not isinstance(inp1, np.ndarray):
        return inp1

    if parameter < 0:
        parameter *= -1

    dist = int(parameter * inp1.shape[0])

    upper = inp1[:dist]
    lower = inp1[dist:]

    shifted = np.append(lower, upper, axis=0)

    return shifted


def shiftUp2(inp1, inp2, parameter):
    """
    Perform a circular left shift on inp2.

    Parameters
    ----------
    inp1 : float or np.ndarray
        First input value. Not actually used by this function.
    inp2 : float or np.ndarray
        Second input value.
    parameter : float
        Determines, how far the image is shifted.

    Return
    ------
    shifted : float or np.ndarray
        Shifted image.
    """

    if not isinstance(inp2, np.ndarray):
        return inp2

    if parameter < 0:
        parameter *= -1

    dist = int(parameter * inp2.shape[0])

    upper = inp2[:dist]
    lower = inp2[dist:]

    shifted = np.append(lower, upper, axis=0)

    return shifted


def shiftDown1(inp1, inp2, parameter):
    """
    Perform a circular downward shift on inp1.

    Parameters
    ----------
    inp1 : float or np.ndarray
        First input value.
    inp2 : float or np.ndarray
        Second input value. Not actually used by this function.
    parameter : float
        Determines, how far the image is shifted.

    Return
    ------
    shifted : float or np.ndarray
        Shifted image.
    """

    if not isinstance(inp1, np.ndarray):
        return inp1

    if parameter < 0:
        parameter *= -1

    dist = inp1.shape[1] - int(parameter * inp1.shape[0])

    upper = inp1[:dist]
    lower = inp1[dist:]

    shifted = np.append(lower, upper, axis=0)

    return shifted


def shiftDown2(inp1, inp2, parameter):
    """
    Perform a circular left shift on inp2.

    Parameters
    ----------
    inp1 : float or np.ndarray
        First input value. Not actually used by this function.
    inp2 : float or np.ndarray
        Second input value.
    parameter : float
        Determines, how far the image is shifted.

    Return
    ------
    shifted : float or np.ndarray
        Shifted image.
    """

    if not isinstance(inp2, np.ndarray):
        return inp2

    if parameter < 0:
        parameter *= -1

    dist = inp2.shape[1] - int(parameter * inp2.shape[0])

    upper = inp2[:dist]
    lower = inp2[dist:]

    shifted = np.append(lower, upper, axis=0)

    return shifted


def shift1(inp1, inp2, parameter):
    """
    Perform a circular left and upwards shift on inp1.

    Parameters
    ----------
    inp1 : float or np.ndarray
        First input value.
    inp2 : float or np.ndarray
        Second input value. Not actually used by this function.
    parameter : float
        Determines, how far the image is shifted.

    Return
    ------
    shifted : float or np.ndarray
        Shifted image.
    """

    if not isinstance(inp1, np.ndarray):
        return inp1

    shifted = shiftLeft1(inp1, inp2, parameter)
    shifted = shiftUp1(inp1, inp2, parameter)

    return shifted


def shift2(inp1, inp2, parameter):
    """
    Perform a circular left and upwards shift on inp2.

    Parameters
    ----------
    inp1 : float or np.ndarray
        First input value. Not actually used by this function.
    inp2 : float or np.ndarray
        Second input value.
    parameter : float
        Determines, how far the image is shifted.

    Return
    ------
    shifted : float or np.ndarray
        Shifted image.
    """

    if not isinstance(inp2, np.ndarray):
        return inp2

    shifted = shiftLeft2(inp1, inp2, parameter)
    shifted = shiftUp2(inp1, inp2, parameter)

    return shifted


def erosion1(inp1, inp2, parameter):
    # if inp1 is not array or inp1 is not 2 dimensional
    if not isinstance(inp1, np.ndarray) or len(inp1.shape) != 2:
        return inp2

    parameter = (parameter + 1) / 2  # convert to [0, 1] range

    ksizex = int(np.round(parameter * inp1.shape[0]))
    if ksizex % 2 == 0:
        ksizex += 1

    ksizey = int(np.round(parameter * inp1.shape[1]))
    if ksizey % 2 == 0:
        ksizey += 1

    kernel = np.ones((ksizex, ksizey), np.uint8)
    erosion = cv2.erode(inp1.astype(np.float32), kernel, iterations=1)

    return erosion


def erosion2(inp1, inp2, parameter):
    return erosion1(inp2, inp1, parameter)


def dilation1(inp1, inp2, parameter):
    # if inp1 is not array or inp1 is not 2 dimensional
    if not isinstance(inp1, np.ndarray) or len(inp1.shape) != 2:
        return inp1

    parameter = (parameter + 1) / 2  # convert to [0, 1] range

    ksizex = int(np.round(parameter * inp1.shape[0]))
    if ksizex % 2 == 0:
        ksizex += 1

    ksizey = int(np.round(parameter * inp1.shape[1]))
    if ksizey % 2 == 0:
        ksizey += 1

    kernel = np.ones((ksizex, ksizey), np.uint8)
    dilation = cv2.dilate(inp1.astype(np.float32), kernel, iterations=1)

    return dilation


def dilation2(inp1, inp2, parameter):
    return dilation1(inp2, inp1, parameter)


def reScale1(inp1, inp2, parameter):
    """
    Downscale inp1 by parameter, then upscale again.

    Parameters
    ----------
    inp1 : float or np.ndarray
        First input value.
    inp2 : float or np.ndarray
        Second input value. Not actually used by this function.
    parameter : float
        Scale factor.

    Return
    ------
    rescaled : float or np.ndarray
        Rescaled image.
    """

    if not isinstance(inp1, np.ndarray) or len(inp1.shape) != 2:
        return inp1

    if parameter < 0:
        parameter *= -1
    if parameter * inp1.shape[0] < 1 or parameter * inp1.shape[1] < 1:
        # we can't resize to size 0
        return inp1

    inp1 = np.float32(inp1)

    scaled = cv2.resize(inp1, (0, 0), fx=parameter, fy=parameter)

    rescaled = cv2.resize(scaled, inp1.shape)

    return rescaled


def reScale2(inp1, inp2, parameter):
    """
    Downscale inp2 by parameter, then upscale again.

    Parameters
    ----------
    inp1 : float or np.ndarray
        First input value. Not actually used by this function.
    inp2 : float or np.ndarray
        Second input value.
    parameter : float
        Scale factor.

    Return
    ------
    rescaled : float or np.ndarray
        Rescaled image.
    """

    return reScale1(inp2, inp1, parameter)


def gabor1(inp1, inp2, parameter):
    """
    Apply a gabor filter to inp1.

    Parameters
    ----------
    inp1 : float or np.ndarray
        First input value.
    inp2 : float or np.ndarray
        Second input value. Not actually used by this function.
    parameter : float
        Used to define the orientation of the filter.

    Return
    ------
    filtered : float or np.ndarray
        Filtered image.
    """

    if not isinstance(inp1, np.ndarray) or len(inp1.shape) != 2:
        return inp1

    if parameter < 0:
        parameter *= -1

    theta = parameter * 180

    kernel = cv2.getGaborKernel((3, 3), 4, theta, 10, 0.5, 0, ktype=cv2.CV_32F)
    # values like in https://cvtuts.wordpress.com/2014/04/27/gabor-filters-a-practical-overview/

    inp1 = np.float32(inp1)
    filtered = cv2.filter2D(inp1, -1, kernel)

    return filtered


def gabor2(inp1, inp2, parameter):
    """
    Apply a gabor filter to inp2.

    Parameters
    ----------
    inp1 : float or np.ndarray
        First input value. Not actually used by this function.
    inp2 : float or np.ndarray
        Second input value.
    parameter : float
        Used to define the orientation of the filter.

    Return
    ------
    filtered : float or np.ndarray
        Filtered image.
    """

    filtered = gabor1(inp2, inp1, parameter)

    return filtered


def resizeThenGabor1(inp1, inp2, parameter):
    """
    Downscale inp1 and then apply a Gabor filter.

    Parameters
    ----------
    inp1 : float or np.ndarray
        First input value.
    inp2 : float or np.ndarray
        Second input value. Not actually used by this function.
    parameter : float
        Used to define the orientation of the filter.

    Return
    ------
    filtered : float or np.ndarray
        Filtered image.
    """

    if not isinstance(inp1, np.ndarray) or len(inp1.shape) != 2:
        return inp1

    if parameter < 0:
        parameter *= -1
    if parameter * inp1.shape[0] < 1 or parameter * inp1.shape[1] < 1:
        # we can't resize to size 0
        return inp1

    inp1 = np.float32(inp1)
    scaled = cv2.resize(inp1, (0, 0), fx=parameter, fy=parameter)

    filtered = gabor1(scaled, inp2, parameter)

    return filtered


def resizeThenGabor2(inp1, inp2, parameter):
    """
    Downscale inp2 and then apply a Gabor filter.

    Parameters
    ----------
    inp1 : float or np.ndarray
        First input value. Not actually used by this function.
    inp2 : float or np.ndarray
        Second input value.
    parameter : float
        Used to define the orientation of the filter.

    Return
    ------
    filtered : float or np.ndarray
        Filtered image.
    """

    filtered = resizeThenGabor1(inp2, inp1, parameter)

    return filtered


def laplace1(inp1, inp2, parameter):
    # if inp1 is not array or inp1 is not 2 dimensional
    if not isinstance(inp1, np.ndarray) or len(inp1.shape) != 2:
        return inp2

    parameter = (parameter + 1) / 2  # convert to [0, 1] range

    ksizex = int(np.round(parameter * inp1.shape[0]))
    if ksizex % 2 == 0:
        ksizex += 1

    ksizey = int(np.round(parameter * inp1.shape[1]))
    if ksizey % 2 == 0:
        ksizey += 1

    kernel = np.ones((ksizex, ksizey), np.uint8)

    return cv2.Laplacian(inp1.astype(np.uint8), cv2.CV_16S, kernel)


def laplace2(inp1, inp2, parameter):
    return laplace1(inp2, inp1, parameter)


def canny1(inp1, inp2, parameter):
    # if inp1 is not array or inp1 is not 2 dimensional
    if not isinstance(inp1, np.ndarray) or len(inp1.shape) != 2:
        return inp2

    parameter = (parameter + 1) / 2  # convert to [0, 1] range

    ksizex = int(np.round(parameter * inp1.shape[0]))
    if ksizex % 2 == 0:
        ksizex += 1

    ksizey = int(np.round(parameter * inp1.shape[1]))
    if ksizey % 2 == 0:
        ksizey += 1

    kernel = np.ones((ksizex, ksizey), np.uint8)

    lower_threshhold = 1
    upper_threshold = 2

    return cv2.Canny(inp1.astype(np.uint8),
                     lower_threshhold,
                     upper_threshold,
                     kernel)


def canny2(inp1, inp2, parameter):
    return canny1(inp2, inp1, parameter)


def localNormalize1(inp1, inp2, parameter):
    """
    Apply local normalization to inp1.

    Parameters
    ----------
    inp1 : float or np.ndarray
        First input value.
    inp2 : float or np.ndarray
        Second input value. Not actually used by this function.
    parameter : float
        Used for gaussian blur.

    Return
    ------
    normalized : float or np.ndarray
        Normalized image.
    """

    if not isinstance(inp1, np.ndarray) or len(inp1.shape) != 2:
        return inp1

    gaussian1 = GaussianBlur(inp1, inp2, parameter)
    intermediate = gaussian1 - inp1

    squared = cv2.pow(intermediate, 2)
    gaussian2 = GaussianBlur(squared, inp2, parameter)
    rooted = cv2.pow(gaussian2, 0.5)

    normalized = intermediate / rooted

    return normalized

def localNormalize2(inp1, inp2, parameter):
    """
    Apply local normalization to inp2.

    Parameters
    ----------
    inp1 : float or np.ndarray
        First input value. Not actually used by this function.
    inp2 : float or np.ndarray
        Second input value.
    parameter : float
        Used for gaussian blur.

    Return
    ------
    normalized : float or np.ndarray
        Normalized image.
    """

    normalized = localNormalize1(inp2, inp1, parameter)

    return normalized

    
def differences1(inp1, inp2, parameter):
    """
    Return the gradient of inp1.

    Parameters
    ----------
    inp1 : float or np.ndarray
        First input value.
    inp2 : float or np.ndarray
        Second input value. Not actually used by this function.
    parameter : float
        Not actually used by this function.

    Return
    ------
    gradient : float or np.ndarray
        Gradient of inp1.
    """

    if not isinstance(inp1, np.ndarray) or len(inp1.shape) != 2:
        return inp1

    gradient = np.gradient(inp1)

    return gradient

def differences2(inp1, inp2, parameter):
    """
    Return the gradient of inp2.

    Parameters
    ----------
    inp1 : float or np.ndarray
        First input value. Not actually used by this function.
    inp2 : float or np.ndarray
        Second input value.
    parameter : float
        Not actually used by this function.

    Return
    ------
    gradient : float or np.ndarray
        Gradient of inp2.
    """

    gradient = differences1(inp2, inp1, parameter)

    return gradient

def avgDifferences1(inp1, inp2, parameter):
    """
    Return average of differences1.

    Parameters
    ----------
    inp1 : float or np.ndarray
        First input value.
    inp2 : float or np.ndarray
        Second input value. Not actually used by this function.
    parameter : float
        Not actually used by this function.

    Return
    ------
    avg : float
        Average of gradient of inp1.
    """

    if not isinstance(inp1, np.ndarray) or len(inp1.shape) != 2:
        return inp1

    gradient = differences1(inp1, inp2, parameter)

    avg = np.mean(gradient)

    return avg

def avgDifferences2(inp1, inp2, parameter):
    """
    Return average of differences2.

    Parameters
    ----------
    inp1 : float or np.ndarray
        First input value. Not actually used by this function.
    inp2 : float or np.ndarray
        Second input value.
    parameter : float
        Not actually used by this function.

    Return
    ------
    avg : float
        Average of gradient of inp2.
    """

    avg = avgDifferences1(inp2, inp1, parameter)

    return avg

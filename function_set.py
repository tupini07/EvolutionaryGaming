import math
######
import types
from functools import wraps

import numpy as np
import scipy
import scipy.stats

import cv2

functions_atari = ["add", "lt", "average", "aminus", "mult", "cmult1", "cmult2", "inv1", "inv2", "abs1", "abs2", "sqrt1", "sqrt2", "cpow1", "cpow2", "ypow", "exp1", "exp2",
                   "sin1", "sin2", "sqrtxy", "acos1", "acos2", "asin1", "asin2", "atan1", "atan2", "stddev", "skew", "kurtosis", "mean", "range", "round_st", "ceil", "floor", "max1", "min1", "max2", "min2"]


functions_openCV = ["GaussianBlur"]

functions = functions_atari + functions_openCV


# np.seterr(all="raise")


def _is_matrix_matrix(inp1, inp2):
    return isinstance(inp1, np.ndarray) and isinstance(inp2, np.ndarray)


def _is_matrix_scalar(inp1, inp2):
    return isinstance(inp1, np.ndarray) and isinstance(inp2, (int, float))


def _is_scalar_matrix(inp1, inp2):
    return isinstance(inp1, (int, float)) and isinstance(inp2, np.ndarray)


def _pad_to_square(inp1):
    size = max(inp1.shape[0], inp1.shape[1])

    padded = np.zeros((size, size))
    padded[:inp1.shape[0], :inp1.shape[1]] = inp1

    return padded


def _pad_matrices(inp1, inp2):
    """
    Pads matrices of different sizes to be the same size

    Parameters
    ----------
    inp1 : np.ndarray
    inp2 : np.ndarray

    Returns
    -------
        [inp1, inp2]
        resized to the same size, by padding the smaller dimensions
    """
    y_size = max(inp1.shape[1], inp2.shape[1])
    x_size = max(inp1.shape[0], inp2.shape[0])

    placeh1 = np.zeros([x_size, y_size])
    placeh2 = np.zeros([x_size, y_size])

    placeh1[:inp1.shape[0], :inp1.shape[1]] = inp1
    placeh2[:inp2.shape[0], :inp2.shape[1]] = inp2

    return placeh1, placeh2


def add(inp1, inp2, parameter, combination_type=None):
    """
    Add to input values.
    If both inputs are arrays, a new array is created, 
    big enough in all dimensions, to hold the values of both arrays.
    E.g., if inp1 has shape (3,6) and inp2 has shape (5,5),
    the output will have shape (5,6).
    The entries not contained in the input arrays are padded with zeros.

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

        # we pad
        m1, m2 = _pad_matrices(inp1, inp2)
        return m1 + m2

    elif _is_matrix_scalar(inp1, inp2):
        return np.average(inp1) + inp2

    elif _is_scalar_matrix(inp1, inp2):
        return inp1 + np.average(inp2)

    else:
        added = inp1 + inp2
        return added


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
        m1, m2 = _pad_matrices(inp1, inp2)
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

    if not isinstance(inp1, np.ndarray):
        return inp1

    ksizex = int(parameter * inp1.shape[0])
    if ksizex % 2 == 0:
        ksizex += 1
    ksizey = int(parameter * inp1.shape[1])
    if ksizey % 2 == 0:
        ksizey += 1

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
    def abs_list(l): return list(map(abs, l))

    if _is_matrix_matrix(inp1, inp2):

        placeh1, placeh2 = _pad_matrices(inp1, inp2)

        diff = placeh1 - placeh2
        abs_arr = np.apply_along_axis(abs_list, 0, diff)

        avg = abs_arr / 2

        return avg

    elif _is_matrix_scalar(inp1, inp2):
        asmatrix2 = np.full(inp1.shape, inp2)
        diff = inp1 - asmatrix2
        abs_arr = np.apply_along_axis(abs_list, 0, diff)

        avg = abs_arr / 2

        return avg

    elif _is_scalar_matrix(inp1, inp2):
        asmatrix1 = np.full(inp2.shape, inp1)
        diff = asmatrix1 - inp2
        abs_arr = np.apply_along_axis(abs_list, 0, diff)

        avg = abs_arr / 2

        return avg

    else:
        return abs(inp1 - inp2) / 2


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
        m1, m2 = _pad_matrices(inp1, inp2)

        return m1 * m2

    else:
        return inp1 * inp2


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
    prod = inp1 * parameter

    return prod


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
    prod = inp2 * parameter

    return prod


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

    if isinstance(inp1, np.ndarray):
        size = max(inp1.shape[0], inp1.shape[1])
        padded = np.zeros((size, size))
        padded[:inp1.shape[0], :inp1.shape[1]] = inp1

        try:
            inv = np.linalg.inv(inp1)
            return inv
        except np.linalg.LinAlgError:
            return inp1
    else:
        inv = 1 / inp1 if inp1 != 0 else 0

        return inv


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

    if isinstance(inp2, np.ndarray):
        size = max(inp2.shape[0], inp2.shape[1])
        padded = np.zeros((size, size))
        padded[:inp2.shape[0], :inp2.shape[1]] = inp2

        try:
            inv = np.linalg.inv(inp2)
            return inv
        except np.linalg.LinAlgError:
            return inp2
    else:
        inv = 1 / inp2 if inp2 != 0 else 0

        return inv


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

    abs1 = abs(inp1)

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

    abs1 = abs(inp1)

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

    abs2 = abs(inp2)

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

    parameter = abs(parameter)
    inp1 = abs(inp1)

    if isinstance(inp1, np.ndarray):

        padded = _pad_to_square(inp1)
        exponent = int(round(parameter))  # exponent must be integer
        cpow1 = np.linalg.matrix_power(padded, exponent)

    else:
        cpow1 = inp1 ** parameter

    return cpow1


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

    parameter = abs(parameter)
    inp2 = abs(inp2)

    if isinstance(inp2, np.ndarray):

        padded = _pad_to_square(inp2)
        exponent = int(round(parameter))  # exponent must be integer
        cpow2 = np.linalg.matrix_power(padded, exponent)

    else:
        cpow2 = inp2 ** parameter

    return cpow2


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

    if _is_matrix_matrix(inp1, inp2):
        inp2 = int(round(abs(np.mean(inp2))))  # exponent must be integer

        padded = _pad_to_square(inp1)
        ypow = np.linalg.matrix_power(padded, inp2)

    elif _is_matrix_scalar(inp1, inp2):
        padded = _pad_to_square(inp1)

        exponent = int(round(inp2))
        ypow = np.linalg.matrix_power(padded, exponent)

    elif _is_scalar_matrix(inp1, inp2):
        exponent = np.mean(inp2)
        ypow = inp1 ** exponent

    else:
        ypow = inp1 ** inp2

    return ypow


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
    exp1 = exp1 - 1
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

    exp2 = np.expm1(inp1)
    exp2 = exp2 - 1
    exp2 = exp2 / (np.exp(1) - 1)

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
        m1, m2 = _pad_matrices(inp1, inp2)

        square1 = np.linalg.matrix_power(m1, 2)
        square2 = np.linalg.matrix_power(m2, 2)

    elif isinstance(inp1, np.ndarray) and isinstance(inp2, float):
        size = max(inp1.shape[0], inp1.shape[1])

        padded1 = np.zeros((size, size))
        padded1[:inp1.shape[0], :inp1.shape[1]] = inp1

        square1 = np.linalg.matrix_power(padded1, 2)
        square2 = inp2 * inp2

    elif isinstance(inp1, float) and isinstance(inp2, np.ndarray):
        size = max(inp2.shape[0], inp2.shape[1])

        padded2 = np.zeros((size, size))
        padded2[:inp2.shape[0], :inp2.shape[1]] = inp2

        square1 = inp1 * inp1
        square2 = np.linalg.matrix_power(padded2, 2)
    else:
        square1 = inp1 * inp1
        square2 = inp2 * inp2

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
    if isinstance(inp1, np.ndarray):
        return inp1.round()

    else:
        return round(inp1)


def ceil(inp1, inp2, parameter):
    return np.ceil(inp1)


def floor(inp1, inp2, parameter):
    return np.floor(inp1)


def max1(inp1, inp2, parameter):
    if isinstance(inp1, np.ndarray):
        return inp1.max()

    else:
        return inp1


def min1(inp1, inp2, parameter):
    if isinstance(inp1, np.ndarray):
        return inp1.min()

    else:
        return inp1


def max2(inp1, inp2, parameter):
    if _is_matrix_matrix(inp1, inp2):
        inp1, inp2 = _pad_matrices(inp1, inp2)

    return np.maximum(inp1, inp2)


def min2(inp1, inp2, parameter):
    if _is_matrix_matrix(inp1, inp2):
        inp1, inp2 = _pad_matrices(inp1, inp2)

    return np.minimum(inp1, inp2)

############


def split_before(inp1, inp2, parameter):
    if isinstance(inp1, np.ndarray):
        idx_p = (parameter + 1) / 2
        split_point = int(round(len(inp1) * idx_p))

        return inp1[:split_point]

    else:
        return inp1


def split_after(inp1, inp2, parameter):
    if isinstance(inp1, np.ndarray):
        idx_p = (parameter + 1) / 2
        split_point = int(round(len(inp1) * idx_p))

        return inp1[split_point:]

    else:
        return inp1


def range_in(inp1, inp2, parameter):
    if isinstance(inp1, np.ndarray):
        if isinstance(inp2, np.ndarray):
            inp2 = inp2.mean()

        inp2 = np.clip(inp2, -1, 1)
        
        right_p = max(inp2, parameter) + 1 / 2
        left_p = min(inp2, parameter) + 1 / 2

        split_right = int(round(len(inp1) * right_p))
        split_left = int(round(len(inp1) * left_p))

        return inp1[split_left:split_right]

    else:
        return inp1


def index_y(inp1, inp2, parameter):
    if isinstance(inp1, np.ndarray):
        if isinstance(inp2, np.ndarray):
            inp2 = inp2.mean()

        inp2 = np.clip(inp2, -1, 1)

        i_p = inp2 + 1 / 2
        split_point = int(round(len(inp1) * i_p))

        return inp1[split_point]

    else:
        return inp1


def index_p(inp1, inp2, parameter):
    if isinstance(inp1, np.ndarray):
        idx_p = (parameter + 1) / 2
        split_point = int(round(len(inp1) * idx_p))

        return inp1[split_point]

    else:
        return inp1


def vectorize(inp1, inp2, parameter):
    if isinstance(inp1, np.ndarray):
        return inp1.flatten()

    else:
        return inp1

def first(inp1, inp2, parameter):
    if isinstance(inp1, np.ndarray):
        return inp1[0]

    else:
        return inp1

def last(inp1, inp2, parameter):
    if isinstance(inp1, np.ndarray):
        return inp1[-1]

    else:
        return inp1

# TODO: differences -> computational derivative of 1D vector of inp1


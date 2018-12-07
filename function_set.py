import numpy as np
import math
import cv2

# TODO: gaussian blur is currently removed. Performance is much worse with it enabled :o 
# might be because the screen size is already very small? blurring 
#functions = ["add", "sinx", "lt"]#, "GaussianBlur"]
functions_atari = ["add", "sinx", "lt", "average", "aminus", "mult", "cmult1", "cmult2", "inv1", "inv2", "abs1", "abs2", "sqrt1", "sqrt2", "cpow1", "cpow2"]
functions_openCV = ["GaussianBlur"]
functions = functions_atari + functions_openCV

def add(inp1, inp2, parameter):
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
    if isinstance(inp1, np.ndarray) and isinstance(inp2, np.ndarray):
        #ugly hack for padding
        
        y_size = max(inp1.shape[1], inp2.shape[1])
        x_size = max(inp1.shape[0], inp2.shape[0])

        placeh1 = np.zeros([x_size, y_size])
        placeh2 = np.zeros([x_size, y_size])

        placeh1[:inp1.shape[0], :inp1.shape[1]] = inp1
        placeh2[:inp2.shape[0], :inp2.shape[1]] = inp2

        return placeh1 + placeh2
        
    elif isinstance(inp1, np.ndarray) and isinstance(inp2, (int, float)):
        return np.average(inp1) + inp2
        
    elif isinstance(inp1, (int, float)) and isinstance(inp2, np.ndarray):
        return inp1 + np.average(inp2)
        
    else:
        added = inp1 + inp2
        return added


def sinx(inp1, inp2, parameter):
    """
    Get sin of first input value.
    If it is an array, the average of its values will be used.

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
    sin : float
        Sin of first input value.
    """

    # TODO: maybe this should return the same np.ndarray, but applying math.sin element-wise?
    if isinstance(inp1, np.ndarray):
        inp1 = np.mean(inp1)
        
    sin = math.sin(inp1)

    return sin


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

    # TODO: maybe this should compare the matrices element-wise? And returns a matrix of the same
    # size as the smaller of the 2 matrices 
    if isinstance(inp1, np.ndarray):
        inp1 = np.mean(inp1)
    if isinstance(inp2, np.ndarray):
        inp2 = np.mean(inp2)

    lesser = min(inp1, inp2)

    return lesser

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

    blurred = cv2.GaussianBlur(inp1, (ksizex,ksizey), 0, sigmaY=0)

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
    abs_list = lambda l: list(map(abs, l))

    if isinstance(inp1, np.ndarray) and isinstance(inp2, np.ndarray):
        #ugly hack for padding
        
        y_size = max(inp1.shape[1], inp2.shape[1])
        x_size = max(inp1.shape[0], inp2.shape[0])

        placeh1 = np.zeros([x_size, y_size])
        placeh2 = np.zeros([x_size, y_size])

        placeh1[:inp1.shape[0], :inp1.shape[1]] = inp1
        placeh2[:inp2.shape[0], :inp2.shape[1]] = inp2

        diff = placeh1 - placeh2
        abs_arr = np.apply_along_axis(abs_list, 0, diff)

        avg = abs_arr / 2

        return avg
        
    elif isinstance(inp1, np.ndarray) and isinstance(inp2, (int, float)):
        asmatrix2 = np.full(inp1.shape, inp2)
        diff = inp1 - asmatrix2
        abs_arr = np.apply_along_axis(abs_list, 0, diff)

        avg = abs_arr / 2

        return avg
        
    elif isinstance(inp1, (int, float)) and isinstance(inp2, np.ndarray):
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

    if isinstance(inp1, np.ndarray) and isinstance(inp2, np.ndarray):
        #ugly hack for padding
        
        y_size = max(inp1.shape[1], inp2.shape[1])
        x_size = max(inp1.shape[0], inp2.shape[0])

        placeh1 = np.zeros([x_size, y_size])
        placeh2 = np.zeros([x_size, y_size])

        placeh1[:inp1.shape[0], :inp1.shape[1]] = inp1
        placeh2[:inp2.shape[0], :inp2.shape[1]] = inp2

        prod = placeh1 * placeh2

        return prod

    else:
        prod = inp1 * inp2

        return prod

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
        padded = np.zeros((size,size))
        padded[:inp1.shape[0], :inp1.shape[1]] = inp1

        try:
            inv = np.linalg.inv(inp1)
            return inv
        except numpy.linalg.LinAlgError:
            return inp1
    else:
        inv = 1 / inp1

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
        padded = np.zeros((size,size))
        padded[:inp2.shape[0], :inp2.shape[1]] = inp2

        try:
            inv = np.linalg.inv(inp2)
            return inv
        except numpy.linalg.LinAlgError:
            return inp2
    else:
        inv = 1 / inp2

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
    Calculate inp1 to the power of parameter.
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
        Square root of inp1.
    """

    parameter = abs(parameter)
    
    if isinstance(inp1, np.ndarray):
        size = max(inp1.shape[0], inp1.shape[1])
        padded = np.zeros((size,size))
        padded[:inp1.shape[0], :inp1.shape[1]] = inp1

        cpow1 = np.linalg.matrix_power(padded, parameter)
    else:
        cpow1 = inp1 ** parameter

    return cpow1

def cpow2(inp1, inp2, parameter):
    """
    Calculate inp2 to the power of parameter.
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
        Square root of inp2.
    """

    parameter = abs(parameter)
    
    if isinstance(inp2, np.ndarray):
        size = max(inp2.shape[0], inp2.shape[1])
        padded = np.zeros((size,size))
        padded[:inp2.shape[0], :inp2.shape[1]] = inp2

        cpow2 = np.linalg.matrix_power(padded, parameter)
    else:
        cpow2 = inp1 ** parameter

    return cpow2

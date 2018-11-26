import numpy as np
import math
import cv2

functions = ["add", "sinx", "lt"]#, "GaussianBlur"]


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

    

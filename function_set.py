import numpy as np

functions = ["add"]


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
        return inp1 + inp2
        

    

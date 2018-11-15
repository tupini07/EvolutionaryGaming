import function_set


class CGP_cell:
    """
    Class representing a cell in a cartesian genetic program.

    Attributes
    ----------
    inputs : (int, int)
        Positions of the cells from which this cell takes its inputs.
    parameter : float
        Extra parameter used by some functions from the function set.
    function : function
        Function from function set used to calculate the output value.
    last_value : float or numpy.ndarray
        Value obtained in last function evaluation.
        Initialized as 0.

    Parameters
    ----------
    genome : [float]
        List of four floating point values between 0 and 1 representing the genome of the cell.
    num_cells : int
        Number of cells in the CGP this cell belongs to. Defaults to 100.
    """

    def __init__(self, genome, num_cells=100):
        #get inputs
        inp1 = round(genome[0] * (num_cells-1))
        inp2 = round(genome[1] * (num_cells-1))
        self.inputs = (inp1, inp2)

        self.parameter = genome[2]

        #get function
        num_functions = len(function_set.functions)
        func_pos = round(genome[3] * (num_functions - 1))
        func_name = functions_set.functions[func_pos]
        self.function = getattr(function_set, func_name)

        self.last_value = 0


    def evaluate(self):
        """
        Evaluates the function of the cell using the inputs and the parameter.

        Returns
        -------
        result : float or numpy.ndarray
            Result of function evaluation.
        """
        func = getattr(function_set, self.function)

        inp1 = self.inputs[0].last_value
        inp2 = self.inputs[1].last_value

        result = function(inp1, inp2, self.parameter)
        self.last_value = result

        return result
        

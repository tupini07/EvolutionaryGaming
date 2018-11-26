import function_set
from CGP_program import CGP_program


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
    program : CGP_Program
        CGP program of which the cell is a part.
    active : boolean
        Marks cell as active.
    last_value : float or numpy.ndarray
        Value obtained in last function evaluation.
        Initialized as 0.

    Parameters
    ----------
    genome : [float]
        List of four floating point values between 0 and 1 representing the genome of the cell.
    program : CGP_Program
        CGP program of which the cell is a part.
    """

    def __init__(self, genome, program):
        #get inputs
        inp1 = round(genome[0] * (program.num_cells-1))
        inp2 = round(genome[1] * (program.num_cells-1))
        self.inputs = (inp1, inp2)

        #get function
        num_functions = len(function_set.functions)
        func_pos = round(genome[2] * (num_functions - 1))
        func_name = functions_set.functions[func_pos]
        self.function = getattr(function_set, func_name)

        self.parameter = genome[3]

        self.program = program

        self.last_value = 0

        self.active = False


    def evaluate(self):
        """
        Evaluates the function of the cell using the inputs and the parameter.

        Returns
        -------
        result : float or numpy.ndarray
            Result of function evaluation.
        """
        func = getattr(function_set, self.function)

        inp1 = self.program.last_value(inputs[0])
        inp2 = self.program.last_value(inputs[1])

        result = function(inp1, inp2, self.parameter)
        self.last_value = result

        return result
        

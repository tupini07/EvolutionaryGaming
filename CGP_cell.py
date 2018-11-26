import function_set
import numpy as np


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
    has_been_evaluated_this_iteration : bool
        Indicates wether this cell has been evaluated at the current generation or not
    is_output_cell : bool
        says if the current cell is an output cell or not

    Parameters
    ----------
    genome : [float]
        List of four floating point values between 0 and 1 representing the genome of the cell.
    program : CGP_Program
        CGP program of which the cell is a part.
    """

    def __init__(self, genome, program):
        # get inputs
        inp1 = round(genome[0] * (program.num_cells-1))
        inp2 = round(genome[1] * (program.num_cells-1))
        self.inputs = (inp1, inp2)

        # get function
        num_functions = len(function_set.functions)
        func_pos = round(genome[2] * (num_functions - 1))
        func_name = function_set.functions[func_pos]
        self.function = getattr(function_set, func_name)

        self.parameter = genome[3]

        self.program = program

        self.last_value = 0

        self.is_output_cell = False
        self.active = False


    def evaluate(self):
        """
        Evaluates the function of the cell using the inputs and the parameter.

        Returns
        -------
        result : float or numpy.ndarray
            Result of function evaluation.
        """

        inp1 = self.program.get_cell(self.inputs[0]).last_value
        inp2 = self.program.get_cell(self.inputs[1]).last_value

        if not self.is_output_cell:
            self.last_value = self.function(inp1, inp2, self.parameter)

        # if it is an output cell then we have no function, so we just take the max input
        else:
            # TODO, all this business about getting the max of the inputs to an ouput node could be 
            # encapsulated somewhere
            final_outputs = [inp1, inp2]

            # convert output nodes to list if they aren't already
            for i, fo in enumerate(final_outputs):
                if not isinstance(fo, list):
                    final_outputs[i] = [fo]

            # get the average value of the lists, this will be the final value of the inputs
            final_outputs = [np.average(ip) for ip in final_outputs]

            # return the max of the inputs
            self.last_value = np.max(final_outputs)


        # return last computed value
        return self.last_value

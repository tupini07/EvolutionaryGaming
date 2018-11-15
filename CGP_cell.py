import function_set


class CGP_cell:
    """
    Class representing a cell in a cartesian genetic program.
    """

    def __init__(self):
        self.inputs = (None, None)
        self.function = None
        self.last_value = 0


    def evaluate(self):
        func = getattr(function_set, self.function)

        inp1 = self.inputs[0].last_value
        inp2 = self.inputs[1].last_value

        self.last_value = function(inp1, inp2)

        return self.last_value
        

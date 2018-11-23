from CGP_cell import CGP_cell

class CGP_program:
    """
    Class representing a cartesion genetic program.

    Attributes
    ----------
    cells : [CGP_cell]
        The cells that make up this program.
    
    Parameters
    ----------
    genome : [[float]]
        Genome encoding representing each cell as a list of floats.
    """

    def __init__(self, genome):
        self.cells = []
        num_cells = len(genome)
        
        for cell_genome in cells:
            cell = CGP_cell(cell_genome, self)
            self.cells.append(cell)

    def last_value(self, cell_num):
        """
        Get last output value of cell indicated by cell_num.

        Parameter
        ---------
        cell_num : int
            Position of cell in the self.cells.

        Return
        ------
        last_value : float or np.ndarray
            Last output of cell.
        """

        last_value = cells[cell_num].last_value

        return last_value

    def evaluate(self, inputs):
        """
        Evaluate all cells given new inputs and produce an output value.

        Parameter
        ---------
        inputs : [np.ndarray]
           Current input split in red, blue, and green.

        Return
        ------
        output : int
           Number of output cell with highest value.
        """

        self.cells[0].last_value = inputs[0]
        self.cells[1].last_value = inputs[1]
        self.cells[2].last_value = inputs[2]

        for cell in self.cells[3:]:
            cell.evaluate()

        outputs = cells[-16:]

        max_val = outputs[0].last_value
        max_index = 0

        for i, output in outputs:
            if output.last_value > max_val:
                max_val = output.last_value
                max_index = i

        return max_index
    

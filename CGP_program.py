import numpy as np
import constants as cc
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
        assert len(genome[0]) == 4, ("Program genome should be a list of lists. The inner lists are the "
                                     "individual cell genomes and should be of length 4")

        self.cells = []
        self.num_cells = len(genome)

        for cell_genome in genome:
            cell = CGP_cell(cell_genome, self)
            self.cells.append(cell)

        self.input_cells = self.cells[:3]
        for c in self.input_cells: # if we specify the output and input cells as different classes then this can be removed
            c.is_input_cell = True

        self.output_cells = self.cells[-cc.N_OUTPUT_NODES:]
        for c in self.output_cells:
            c.is_output_cell = True

        self.mark_active()


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

        return self.get_cell(cell_num).last_value

    def get_cell(self, cell_num) -> CGP_cell:
        """
        Just returns the cell object indicated by cell_num

        cell_num : int
            Position of cell in the self.cells.

        Return
        ------
        cell : CGP_cell
            Instance of the cell at index `num_cell`.
        """
        return self.cells[cell_num]


    def evaluate(self, inputs):
        """
        Evaluate all cells given new inputs and produce an output value.
        The output value here is the number of the action [0,15] that we 
        want to perform in the atari game 

        Parameter
        ---------
        inputs : [np.ndarray]
           Current input split in red, blue, and green.

        Return
        ------
        output : int
           Number of output cell with highest value.
        """

        # store new inputs
        for i, input_cell in enumerate(self.input_cells):
            input_cell.last_value = inputs[i]

        # evaluate every cell that is not an input cell
        for cell in self.cells[3:]:
            if cell.active:
                cell.evaluate()

        # just get the index of the "ouput" which has the maximum value
        outputs = [o.last_value for o in self.output_cells]

        max_index = np.argmax(outputs)

        return max_index
    

    def mark_active(self):
        """
        Mark cells whose output isn't connected to the outputs as inactive
        and the other cells as active.
        """
        queue = self.cells[-cc.N_OUTPUT_NODES:] #put outputs in queue
        found = set(queue)

        while queue != []:
            curr = queue[0]
            queue = queue[1:]

            curr.active = True

            i1, i2 = curr.inputs
            
            cell1 = self.cells[i1]
            if not cell1 in found:
                queue.append(cell1)
                found.add(cell1)

            cell2 = self.cells[i2]
            if not cell2 in found:
                queue.append(cell2)
                found.add(cell2)

            
            
            

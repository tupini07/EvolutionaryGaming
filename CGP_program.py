import os

import numpy as np
from graphviz import Digraph

import constants as cc
from CGP_cell import CGP_cell, Output_cell


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
        self.genome = genome

        self.cells = []
        self.num_cells = len(genome)

        for cell_genome in genome[:-cc.N_OUTPUT_NODES]:
            cell = CGP_cell(cell_genome, self)
            self.cells.append(cell)

        for cell_genome in genome[-cc.N_OUTPUT_NODES:]:
            cell = Output_cell(cell_genome, self)
            self.cells.append(cell)

        self.input_cells = self.cells[:3]

        self.output_cells = self.cells[-cc.N_OUTPUT_NODES:]
        
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
        queue = self.cells[-cc.N_OUTPUT_NODES:]  # put outputs in queue
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

    def draw_function_graph(self, picture_name="gcp_program_graph"):
        """
        Renders CGP program as an svg image. The image is saved to disk,
        
        Parameters
        ----------
        picture_name : str, optional
            the name we want to give to the image file (the default is "gcp_program_graph")
        
        """
        dot = Digraph(name='GCP Program', format='svg')

        with dot.subgraph(name="clusterInputs") as inputs, dot.subgraph(name="clusterOutputs") as outputs:

            # add every cell
            for ic, cell in enumerate(self.cells):
                if cell.active:
                    if cell in self.input_cells:
                        inputs.node(str(ic), "Input " + str(ic))

                    elif cell in self.output_cells:
                        outputs.node(str(ic), "Output " + str(ic))

                    else:
                        dot.node(str(ic), cell.function.__name__)

        for ic, cell in enumerate(self.cells):
            if not cell.active or ic <= 2:
                continue

            for inp in cell.inputs:
                dot.edge(str(inp), str(ic))

        dot.render(picture_name)
        os.remove(picture_name)  # remove RAW dot file

    def __repr__(self):
        """
        Returns the string representation of the GCP program. This representation can then be used to recreate
        the GCP program by using the `from_repr` class method.
        """

        return "GCP_Program:Genome::" + str(self.genome)

    @classmethod
    def from_repr(cls, repr):
        """
        Given a string representation of a GCP program, return an instance of GCP_Program which corresponds
        to the representation. 

        Parameters
        ----------
        repr : string
            The representation of the GCP_Program

        """
        if type(repr) == str:
            repr = eval(repr.split("::")[1])

        return cls(repr)

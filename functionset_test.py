import numpy as np
import pytest

import function_set as fst

# These are just simple tests to check that the functions in the functionset
# behave as expected
# These are to be executed with pytest


def test_functionset_accept_type():
    """
    All functions in function set should work with the following combination of inputs
        np.ndarray, np.ndarray
        Scalar, np.ndarray
        np.ndarray, Scalar
        Scalar, Scalar 
    """

    for fname in fst.functions:
        if fname.startswith("_"):
            continue
        
        func = getattr(fst, fname)

        test_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        test_scalar = 42
        test_parameter = 0.5

        tests = [[test_matrix, test_matrix],
                 [test_scalar, test_matrix],
                 [test_matrix, test_scalar],
                 [test_scalar, test_scalar]]

        for t_inp1, t_inp2 in tests:
            try:
                func(t_inp1, t_inp2, test_parameter)
            except Exception:
                pytest.fail(f"Function '{fname}' seems to fail when the inputs are ({type(t_inp1)}, {type(t_inp2)})")
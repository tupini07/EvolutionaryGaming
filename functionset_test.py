import random
import sys

import numpy as np
import pytest

import constants as cc
import evolutionary_problem as eap
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
        test_matrix2 = np.array([[11, 22, 33, 44, 41], [55, 66, 77, 88, 85], [
                                99, 100, 111, 112, 199], [113, 114, 115, 116, 133]])
        test_scalar = 42
        test_parameter = 0.5

        tests = [[test_matrix, test_matrix],
                 [test_matrix, test_matrix2],
                 [test_matrix2, test_matrix],
                 [test_matrix2, test_matrix2],
                 [test_scalar, test_matrix],
                 [test_matrix, test_scalar],
                 [test_scalar, test_scalar],
                 [0, 0]]

        for t_inp1, t_inp2 in tests:
            try:
                res = func(t_inp1, t_inp2, test_parameter)

                if res is None:
                    pytest.fail(
                        f"it seems that function '{fname}' does not return a value.")

                if isinstance(res, np.ndarray) and len(res) == 0:
                    pytest.fail(
                        f"it seems that function '{fname}' returns an empty array for inputs of types ({type(t_inp1)}, {type(t_inp2)}). "
                        f"With the respective shapes {np.array(t_inp1).shape} and {np.array(t_inp2).shape}")

            except Exception as err:
                pytest.fail(
                    f"Function '{fname}' seems to fail when the inputs are ({type(t_inp1)}, {type(t_inp2)}) "
                    f"With the respective shapes {np.array(t_inp1).shape} and {np.array(t_inp2).shape}"
                    f"\nwith the following exception:\n{err}")


def test_individual_generation_size():
    if len(eap.generator(random.Random(), {})) != cc.N_EVOLVABLE_GENES:
        pytest.fail(
            "The generator is not generating individuals of the correct size.")


def test_mutation_same_size():
    class ea_test():
        bounder = eap.bounder

    individual = eap.generator(random.Random(), {})

    before_mutation = len(individual)

    individual = eap.mutate(random.Random(), [individual], {
                            "mutation_rate": 2,
                            "_ec": ea_test})[0]

    after_mutation = len(individual)

    if before_mutation != after_mutation:
        pytest.fail(
            f"The length before mutation ({before_mutation}) is not the same as after mutation ({after_mutation})")


def test_length_of_returns():
    """
    execute pytest like this:
        pytest -s

    to see the print statements
    """
    print()
    test_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    tests = [[test_matrix, test_matrix]]
    test_parameter = 0.5

    print(f"input shape is {test_matrix.shape}")

    if "-s" not in sys.argv:
        return

    for fname in fst.functions:
        if fname.startswith("_"):
            continue

        func = getattr(fst, fname)

        for t_inp1, t_inp2 in tests:
            try:
                res = func(t_inp1, t_inp2, test_parameter)

                print(
                    f"{fname}({type(t_inp1)}, {type(t_inp2)}) => {np.array(res).shape}")

            except Exception:
                pass

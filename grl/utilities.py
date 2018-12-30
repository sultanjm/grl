import numpy as np
import enum

def sample(p_row):
    return np.random.choice(len(p_row), p=p_row)

def generateRandomProbabilityMatrix(dim1=2, dim2=4, repeat_second_dimension=True):
    '''
    usage-1: generateRandomProbabilityMatrix(num_of_actions, num_of_states) to generate a transition matrix.
    usage-2: generateRandomProbabilityMatrix(num_of_states, num_of_percepts, False) to generate an emission matrix.
    '''
    if repeat_second_dimension:
        T = np.random.rand(dim1, dim2, dim2)
    else:
        T = np.random.rand(dim1, dim2)
    # WARNING! This code is not robust as, rarely, all the elements in a row can be zero
    T = T / T.sum(axis=-1, keepdims=True)
    return T

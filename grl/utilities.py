import numpy as np
import enum
import grl

__all__ = ['sample', 'random_probability_matrix', 'epsilon_sample', 'optimal_policy']

def sample(p_row):
    return np.random.choice(len(p_row), p=p_row)

def random_probability_matrix(dim1=2, dim2=4, repeat_second_dimension=True):
    '''
    usage-1: random_probability_matrix(num_of_actions, num_of_states) to generate a transition matrix.
    usage-2: random_probability_matrix(num_of_states, num_of_percepts, False) to generate an emission matrix.
    '''
    if repeat_second_dimension:
        T = np.random.rand(dim1, dim2, dim2)
    else:
        T = np.random.rand(dim1, dim2)
    # WARNING! This code is not robust as, rarely, all the elements in a row can be zero
    T = T / T.sum(axis=-1, keepdims=True)
    return T

def epsilon_sample(vect, argmax=None, epsilon=1.0):
    p = np.random.random()
    if p > epsilon:
        return argmax
    else:
        return vect[np.random.choice(len(vect))]   

def optimal_policy(Q):
    if not isinstance(Q, grl.learning.Storage) or Q.dimensions != 2:
        raise RuntimeError("No valid Storage object is provided.")
    policy = {}
    # needs fixing: storage should not be used out side of Storage
    for state in Q.storage.keys():
        policy[state] = max(Q[state])[1]
    return policy 

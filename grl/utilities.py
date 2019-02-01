import numpy as np
import enum
import grl
import random


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
    p = random.uniform(0,1)
    if p > epsilon:
        return argmax
    else:
        return random.sample(vect, 1)[0]  

def optimal_policy(Q):
    if not isinstance(Q, grl.learning.Storage) or Q.dimensions != 2:
        raise RuntimeError("No valid Storage object is provided.")
    policy = {}
    for state in Q.keys():
        policy[state] = max(Q[state])[1]
    return policy 

def bits2int(bits):
    v = 0
    for b in bits:
        v = (v << 1) | b
    return v

def int2bits(n):
    return [1 if b=='1' else 0 for b in bin(n)[2:]]

def occurrence_ratio_processor(storage, key, event, min_ratio=0.0):
    h = event.data['h']
    update = event.data['update']
    stats = h.stats.get(storage, dict())
    update_t = len(update)/h.steplen
    
    prob = stats.get(key, min_ratio)
    if event.type == grl.EventType.ADD:
        stats[key] = (prob*h.t + update.count(key))/(update_t + h.t)
    if event.type == grl.EventType.REMOVE:
        stats[key] = (prob*(h.t + update_t) - update.count(key))/h.t
    
    stats[key] = max(stats[key], min_ratio)
    
    h.stats[storage] = stats

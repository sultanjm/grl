import collections
import numpy as np
import grl

class HistoryManager:

    def __init__(self, history=[], start_timestep=0, MAX_LENGTH=None, history_to_state_map=lambda h: h):
        self.MAX_LENGTH = MAX_LENGTH
        self.h = collections.deque(history, MAX_LENGTH)
        self.t = start_timestep + len(history)
        self.h2s_map = history_to_state_map
        self.part = []
        
    def getHistory(self):
        return self.h

    def getTimestep(self):
        return self.t

    def getPartialUpdate(self):
        return self.part
    
    def setHistory(self, history, start_timestep=0):
        self.h.clear()
        self.h.append(history)
        self.t = start_timestep + len(history)

    def extendHistory(self, elem, is_complete=True):
        # the type of elem can be different for each manager
        # in a standard grl framework, it is a tuple of (a,e)
        self.part.append(elem)
        if is_complete:
            elem = tuple(self.part)
            self.part.clear()
            self.h.append(elem)
            self.t += 1
    
    def getMappedState(self, h=None):
        if not h:
            h = self.h
        return self.h2s_map(h)

class PerceptManager:

    def __init__(self, emission_func=lambda s : s, label_func=lambda s: s, inv_label_func=lambda s: s):
        self.E = emission_func
        self.l = label_func
        self.inv_l = inv_label_func

    def setEmissionFunction(self, emission_func):
        self.E = emission_func
    
    def setLabelFunctions(self, label_func, inv_label_func):
        self.l = label_func
        self.inv_l = inv_label_func

    def getLabelFunctions(self):
        return (self.l, self.inv_l)

    def getEmissionFunction(self):
        return self.E

    def getPercept(self, state_l):
        return self.l(self.perception(self.inv_l(state_l)))

    def perception(self, state):
        # WARNING! This part of the code is not robust enough.
        # assuming a 'valid' emission function
        if callable(self.E):
            e = self.inv_l(self.E(self.l(state)))
        # assuming a 'valid' emission kernel
        elif len(np.shape(self.E)) == 2:
            e = grl.utilities.sample(self.E[state,:])
        else:
            raise RuntimeError("No valid percept function is provided.")
        return e


class StateManager:
    
    def __init__(self, transition_func=lambda s,a: s, start_state_l=0, label_func=lambda s: s, inv_label_func=lambda s: s):
        self.T = transition_func
        self.l = label_func
        self.inv_l = inv_label_func
        # WARNING! This part of the code is not robust enough.
        # assuming 0 is a 'valid' labled state
        self.s = self.inv_l(start_state_l)
        
    def getCurrentState(self):
        return self.l(self.s)

    def setCurrentState(self, state_l):
        self.s = self.inv_l(state_l)

    def setLabelFunctions(self, label_func, inv_label_func):
        self.l = label_func
        self.inv_l = inv_label_func
    
    def getLabelFunctions(self):
        return (self.l, self.inv_l)

    def setTransitionFunction(self, transition_func):
        self.T = transition_func

    def getTransitionFunction(self):
        return self.T

    def getNextState(self, action, state_l):
        return self.l(self.simulate(action, self.inv_l(state_l)))

    def simulate(self, action, state=None):
        if not state:
            state = self.s
        # WARNING! This part of the code is not robust enough.
        # assuming a 'valid' transition function
        if callable(self.T): 
            s = self.inv_l(self.T(self.l(state), action))
        # assuming a 'valid' probability kernel
        elif len(np.shape(self.T)) == 3: 
            s = grl.utilities.sample(self.T[action,state,:])
        else:
            raise RuntimeError("No valid transition function is provided.")
        return s

    def transit(self, action):
        self.s = self.simulate(action)
        return self.s


class ActionManager:

    def __init__(self, actions_l=[0], label_func=lambda a: a, inv_label_func=lambda a: a):
        self.l = label_func
        self.inv_l = inv_label_func
        self.actions_l = actions_l
        self.actions = [self.inv_l(a_l) for a_l in actions_l]

    def getActions(self):
        return self.actions_l

    def setActions(self, actions_l):
        self.actions_l = actions_l
        self.actions = [self.inv_l(a_l) for a_l in actions_l]


class RewardManager:

    def __init__(self, reward_func=lambda s_l, a_l, s_nxt_l, e_nxt_l, h: 0):
        self.r_func = reward_func

    def setRewardFunction(self, reward_func):
        self.r_func = reward_func
    
    def getRewardFunction(self):
        return self.r_func

    def r(self, s_l, a_l, s_nxt_l=None, e_nxt_l=None, h=None):
        return self.r_func(s_l, a_l, s_nxt_l, e_nxt_l, h)
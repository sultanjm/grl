import collections
import numpy as np
import grl

class HistoryManager:

    def __init__(self, history=[], start_timestep=0, MAX_LENGTH=None, state_map=lambda h: h):
        self.MAX_LENGTH = MAX_LENGTH
        self.history = collections.deque(history, MAX_LENGTH)
        self.t = start_timestep + len(history)
        self.state_map = state_map
        self.partial_extension = []
    
    def set_history(self, history, start_timestep=0):
        self.history.clear()
        self.history.append(history)
        self.t = start_timestep + len(history)

    def extend_history(self, elem, complete=True):
        # the type of elem can be different for each manager
        # in a standard grl framework, it is a tuple of (a,e)
        self.partial_extension.append(elem)
        if complete:
            elem = tuple(self.partial_extension)
            self.partial_extension.clear()
            self.history.append(elem)
            self.t += 1
    
    def mapped_state(self, h=None):
        if not h:
            h = self.history
        return self.state_map(h)

class PerceptManager:

    def __init__(self, emission_func=lambda s : s, percepts=None, percept=None):
        self.emission_func = emission_func
        self.percepts = percepts
        self.percept = percept

    def perception(self, state):
        if not callable(self.emission_func):
            raise RuntimeError("No valid percept function is provided.")      
        return self.emission_func(state)


class StateManager:
    
    def __init__(self, transition_func=lambda s,a: s, start_state=None, states=None):
        self.transition_func = transition_func
        self.state = start_state
        self.states = states

    def simulate(self, action, state=None):
        if not state:
            state = self.state
        if not callable(self.transition_func): 
            raise RuntimeError("No valid transition function is provided.")
        return self.transition_func(state, action)

    def transit(self, action):
        if action:
            self.state = self.simulate(action)
        return self.state


class ActionManager:

    def __init__(self, actions=None, action=None):
        self.actions = actions
        self.action = action


class RewardManager:

    def __init__(self, reward_func=lambda h, a, e_nxt, s, s_nxt: 0):
        self.reward_func = reward_func

    def r(self, h, a, e_nxt, s=None, s_nxt=None):
        return self.reward_func(h, a, e_nxt, s, s_nxt)
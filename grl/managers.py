import collections
import numpy as np
import grl

__all__ = ['HistoryManager', 'StateManager', 'PerceptManager', 'ActionManager', 'RewardManager']

class HistoryManager:
    def __init__(self, state_map=lambda a, e, h: None, history=[], *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.history = collections.deque(history, self.kwargs.get('maxlen', None))
        self.t = len(history) + self.kwargs.get('start_timestep', 0)
        self.state_map = state_map

        # internal parameters
        self.partial_extension = []
    
    def set_history(self, history, start_timestep=0):
        self.history.clear()
        self.history.append(history)
        self.t = start_timestep + len(history)

    def extend_history(self, elem, complete=True):
        # the type of elem can be different for each manager
        # in a standard grl framework, it is a tuple of (a,e)
        if complete:
            if self.partial_extension:
                elem = tuple(self.partial_extension)
                self.partial_extension.clear()
            self.history.append(elem)
            self.t += 1
        else:
            self.partial_extension.append(elem)
    
    def mapped_state(self, a=None, e=None, h=None):
        if not h:
            h = self.history
        return self.state_map(a, e, h)

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
        self.prev_state = None
        self.states = states

    def simulate(self, action, state=None):
        if not state:
            state = self.state
        if not callable(self.transition_func): 
            raise RuntimeError("No valid transition function is provided.")
        return self.transition_func(state, action)

    def transit(self, action):
        if action:
            self.prev_state = self.state
            self.state = self.simulate(action)
        return self.state


class ActionManager:
    def __init__(self, actions=None, action=None):
        self.actions = actions
        self.action = action


class RewardManager:
    def __init__(self, reward_func=lambda a, e, h: 0):
        self.reward_func = reward_func

    def r(self, a, e, h):
        return self.reward_func(a, e, h)
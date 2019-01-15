import collections
import itertools
import numpy as np
import grl
import copy

__all__ = ['History', 'HistoryManager', 'StateManager', 'PerceptManager', 'ActionManager', 'RewardManager']

class History(collections.MutableSequence):

    def __init__(self, *args, **kwargs):
        self.history = collections.deque(*args, **kwargs)
        self.extension = list()
        self.time = 0
        self.xtime = 0

    @property
    def t(self):
        return self.time + self.xtime

    @t.setter
    def t(self, time):
        if not isinstance(time, collections.Sequence):
            time = [time, self.xtime]
        self.time, self.xtime = time

    # TODO: support slicing
    def __getitem__(self, index):
        storage, index = self.__adjust_index__(index)
        return storage[index]

    def __iter__(self):
        for v in self.history:
            yield v
        for v in self.extension:
            yield v

    def insert(self, index, value):
        # TODO: a patchy implementation
        self.history.append(value)
    
    def __delitem__(self, index):
        storage, index = self.__adjust_index__(index)
        del storage[index]

    def __setitem__(self, index, value):
        storage, index = self.__adjust_index__(index)
        storage[index] = value
        
    def __len__(self):
        return len(self.history) + len(self.extension)

    def __repr__(self):
        return repr(self.history) + ' || ' + repr(self.extension)

    def __adjust_index__(self, index):
        if index < 0:
            if -index < len(self.extension) + 1:
                return self.extension, index
            else:
                return self.history, index + len(self.extension)
        else:
            if index > len(self.history) - 1:
                return self.extension, index - len(self.history)
            else:
                return self.history, index


class HistoryManager:
    def __init__(self, state_map=lambda hm, *x, **y: '<?>', history=[], *args, **kwargs):
        self.maxlen = kwargs.get('maxlen', None)
        self.history = History(history, maxlen=self.maxlen)
        self.history.t = kwargs.get('timestep', 0)
        self.state_map = state_map
        self.listeners = collections.defaultdict(set)

    def extend(self, *args):
        for arg in args:
            self.history.extend(arg)
        if args: self.history.t += 1
        return self

    @property
    def h(self):
        return self.history

    @h.setter
    def h(self, history):
        if not isinstance(history, History):
            self.history = History(history, maxlen=self.maxlen)
        else:
            self.history = history

    # def extend_history(self, elem, complete=True):
    #     # the type of elem can be different for each manager
    #     # in a standard grl framework, it is a tuple of (a,e)
    #     if complete:
    #         if self.partial_extension:
    #             elem = tuple(self.partial_extension)
    #             self.partial_extension.clear()
    #         self.history.append(elem)
    #         self.t += 1
    #     else:
    #         self.partial_extension.append(elem)
    
    def state(self, history, extension, level, *args, **kwargs):
        hm = self.assert_manager(history)
        hm.request.clear()
        hm.request['level'] = level
        hm.request['extension'] = extension
        hm.request['args'] = args
        hm.request['kwargs'] = kwargs
        s = self.state_map(hm, *args, **kwargs)
        hm.request.clear()
        return s


    def previous_state(self, h=None, *args, **kwargs):
        pass

    def assert_manager(self, history):
        hm = self
        if history is not self.history:
            hm = copy.deepcopy(self)
            hm.set_history(history)
        return hm

    def register(self, obj, event_name='update'):
        self.listeners[event_name].add(obj)
    
    def deregister(self, obj, event_name='update'):
        self.listeners[event_name].discard(obj)
    
    def dispatch(self, event):
        for obj in self.listeners[event['name']]:
            obj.on(event)
   
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
        if state is None:
            state = self.state
        if not callable(self.transition_func): 
            raise RuntimeError("No valid transition function is provided.")
        return self.transition_func(state, action)

    def transit(self, action):
        if action is not None:
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
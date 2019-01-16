import collections
import itertools
import numpy as np
import grl
import copy

__all__ = ['History', 'HistoryManager', 'StateManager', 'PerceptManager', 'ActionManager', 'RewardManager']

class History(collections.MutableSequence):

    def __init__(self, *args, **kwargs):
        self.history = collections.deque(kwargs.get('history', list()), kwargs.get('maxlen', None))
        self.extension = list()
        self.steps = 0.0
        self.xsteps = 0.0
        self.steplen = kwargs.get('steplen', 1)
        self.stats = kwargs.get('stats', dict())

    @property
    def t(self):
        return self.steps + self.xsteps

    # TODO: no slicing at the moment
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
        return repr(self.history) + ' || ' + repr(self.extension) + ' <steps={},xsteps={},steplen={}>'.format(self.steps, self.xsteps, self.steplen)

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
        self.steplen = kwargs.get('steplen', 2)
        self.maxlen = None if not kwargs.get('maxlen', None) else self.steplen * kwargs.get('maxlen', None)
        self.history = History(history=history, maxlen=self.maxlen, steplen=self.steplen)
        self.history.steps = kwargs.get('steps', 0.0)
        self.state_map = state_map
        self.listeners = collections.defaultdict(set)

    def record(self, items):
        steps = len(items) / self.steplen
        for item in items:
            self.history.append(item)
        self.history.steps += steps
        return self

    def extension(self, items):
        steps = len(items) / self.steplen
        for item in items:
            self.history.extension.append(item)
        self.history.xsteps += steps
        return self

    def drop(self, steps=1.0):
        if not (float(steps) * self.steplen).is_integer():
            raise RuntimeError("unable to drop {} elements.".format(steps * self.steplen))
        dropped = list()
        for _ in range(int(steps*self.steplen)):
            dropped.append(self.history.pop())
        self.history.steps -= steps
        return dropped[::-1]

    def xdrop(self, steps=None):
        if steps is None: steps = self.history.xsteps
        dropped = list()
        if not (float(steps) * self.steplen).is_integer():
            raise RuntimeError("unable to drop {} elements.".format(steps * self.steplen))
        for _ in range(int(steps*self.steplen)):
            dropped.append(self.history.extension.pop())
        self.history.xsteps -= steps
        return dropped[::-1]

    def xmerge(self):
        if self.history.extension:
            for arg in self.history.extension:
                self.history.append(arg)
            self.history.steps += self.history.xsteps
            self.xdrop(self.history.xsteps)
        else:
            raise ValueError("The extension is empty.")

    @property
    def h(self):
        return self.history

    @h.setter
    def h(self, history):
        if not isinstance(history, History):
            raise RuntimeError("No valid History object is provided.")
        self.history = history

    def state(self, history, *args, **kwargs):
        hm = self.assert_hm(history)
        extension = kwargs.get('extension', list())
        level = kwargs.get('level', 'current')
        dropped_h = list()
        dropped_xtn = list()

        if level == 'next':
            dropped_xtn = hm.xdrop()
            hm.extension(dropped_xtn).extension(extension)
        elif level == 'previous':
            dropped_xtn = hm.xdrop(1.0)
            if not dropped_xtn:
                dropped_h = hm.drop(1.0)

        s = self.state_map(history, *args, **kwargs)

        hm.xdrop()
        hm.record(dropped_h).extension(dropped_xtn)

        return s

    def assert_hm(self, history):
        hm = self
        if history is not self.history:
            hm = copy.deepcopy(self)
            hm.history = history
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
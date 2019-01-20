import collections
import itertools
import numpy as np
import grl
import copy
import enum

__all__ = ['History', 'Index', 'HistoryManager', 'StateManager', 'PerceptManager', 'ActionManager', 'RewardManager']

class Index(enum.Enum):
    NEXT = enum.auto()
    CURRENT = enum.auto()
    RECENT = enum.auto()
    PREVIOUS = enum.auto()
    OLDEST = enum.auto()
    FIRST = enum.auto()
    ABSOLUTE = enum.auto()

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
        return repr(self.history) + ' | ' + repr(self.extension) + ' <steps={},xsteps={},steplen={}>'.format(self.steps, self.xsteps, self.steplen)

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

    def extract(self, order, index=Index.CURRENT):
        # TODO: only CURRENT and PREVIOUS extractions are supported    
        assert(order > 0 and order <= self.steplen)
        order = order % self.steplen
        head = int((self.t % 1) * self.steplen)
        idx = order - head - 1
        if order > head: idx -= self.steplen
        if index == Index.PREVIOUS: idx -= self.steplen
        return self[idx]

class HistoryManager:

    def __init__(self, state_map=lambda *x, **y: '<?>', history=[], *args, **kwargs):
        self.steplen = kwargs.get('steplen', 2)
        self.maxlen = None if not kwargs.get('maxlen', None) else self.steplen * kwargs.get('maxlen', None)
        self.history = History(history=history, maxlen=self.maxlen, steplen=self.steplen)
        self.history.steps = kwargs.get('steps', 0.0)
        self.state_map = state_map
        self.listeners = collections.defaultdict(set)

    def record(self, items, notify=True):
        steps = len(items) / self.steplen
        if steps and notify: 
            self.dispatch(grl.EventType.ADD, {'h':self.history, 'update':items})
        for item in items:
            self.history.append(item)
        self.history.steps += steps

        return self

    def extend(self, items, notify=True):
        steps = len(items) / self.steplen
        if steps and notify:
            self.dispatch(grl.EventType.ADD, {'h':self.history, 'update':items})
        for item in items:
            self.history.extension.append(item)
        self.history.xsteps += steps
        return self

    def drop(self, steps=1.0, notify=True):
        if not (float(steps) * self.steplen).is_integer():
            raise RuntimeError("unable to drop {} elements.".format(steps * self.steplen))
        dropped = list()
        for _ in range(int(steps*self.steplen)):
            dropped.append(self.history.pop())
        self.history.steps -= steps
        if dropped and notify:
            self.dispatch(grl.EventType.REMOVE, {'h':self.history, 'update':dropped[::-1]})
        return dropped[::-1]

    def xdrop(self, steps=None, notify=True):
        if steps is None: steps = self.history.xsteps
        dropped = list()
        if not (float(steps) * self.steplen).is_integer():
            raise RuntimeError("unable to drop {} elements.".format(steps * self.steplen))
        for _ in range(int(steps*self.steplen)):
            dropped.append(self.history.extension.pop())
        self.history.xsteps -= steps
        if dropped and notify: 
            self.dispatch(grl.EventType.REMOVE, {'h':self.history, 'update':dropped[::-1]})
        return dropped[::-1]

    def xmerge(self):
        if self.history.extension:
            self.record(self.xdrop(notify=False), notify=False)
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
        index = kwargs.get('index', Index.CURRENT)
        extension = kwargs.get('extension', list())
        hm = self.assert_hm(history)

        change = hm.amend(history, index, extension)
        s = self.state_map(history, *args, **kwargs)
        hm.mend(change)
        return s

    def assert_hm(self, history):
        hm = self
        if history is not self.history:
            hm = copy.deepcopy(self)
            hm.history = history
        return hm

    def register(self, obj, event_type=grl.EventType.ALL):
        self.listeners[event_type].add(obj)
    
    def deregister(self, obj, event_type=grl.EventType.ALL):
        self.listeners[event_type].discard(obj)
    
    def dispatch(self, event_type, data):
        evt = grl.Event(event_type, data)
        for obj in self.listeners[event_type]:
            obj.on(evt)

    def amend(self, h, index=Index.CURRENT, extension=list()):
        old_h = list()
        old_xtn = list()
        # prepare history for non-current indexes
        if index == Index.NEXT:
            old_xtn = self.xdrop()
            self.extend(old_xtn).extend(extension)
        elif index == Index.PREVIOUS:
            old_xtn = self.xdrop(1.0)
            if not old_xtn:
                old_h = self.drop(1.0)
        return [old_h, old_xtn]

    def mend(self, change):
        old_h, old_xtn = change
        # undo the changes in the history
        self.xdrop()
        self.record(old_h).extend(old_xtn)
        return self


class PerceptManager:

    def __init__(self, emission_func=lambda s : s, percept_space=None, e=None):
        self.emission_func = emission_func
        self.percept_space = percept_space
        self.percept = e

    def perception(self, s):
        if not callable(self.emission_func):
            raise RuntimeError("No valid percept function is provided.")      
        return self.emission_func(s)


class StateManager: 

    def __init__(self, transition_func=lambda s,a: s, state_space=None, *args, **kwargs):
        self.transition_func = transition_func
        self.hm = HistoryManager(maxlen=kwargs.get('max_history', 1), steplen=1)
        self.state = kwargs.get('start_state', None)
        self.state_space = state_space

    def simulate(self, a, s=None):
        if s is None:
            s = self.state
        if not callable(self.transition_func): 
            raise RuntimeError("No valid transition function is provided.")
        return self.transition_func(s, a)

    def transit(self, a):
        if a is not None:
            self.hm.record([self.state])
            self.state = self.simulate(a)
        return self.state


class ActionManager:

    def __init__(self, action_space=None, a=None):
        self.action_space = action_space
        self.action = a
    
class RewardManager:

    def __init__(self, reward_func=lambda h, *x, **y: 0):
        self.reward_func = reward_func
        self.hm = grl.HistoryManager()

    def r(self, h, *args, **kwargs):
        self.hm.h = h
        extension = kwargs.get('extension', list())
        index = kwargs.get('index', Index.CURRENT)
        change = self.hm.amend(h, index, extension)

        reward = self.reward_func(h)

        self.hm.mend(change)
        return reward
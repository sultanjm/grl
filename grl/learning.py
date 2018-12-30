import grl
import collections
import numpy as np

class Storage(collections.MutableMapping):

    # MAX, ARGMAX, MIN, ARGMIN, USER_FUNC (v2)
    # >>> f[s].argmax()
    # 'a1'
    # >>> f[s].min()
    # 0
    # >>> f[s].user_func()

    # the user function must have an incremental update structure
    # e.g. user_func_max(v)
    # 
    # if v > old_max:
    #   old_max = v

    def __init__(self, data=None, default_value_range=(0,1)):
        self.storage = collections.defaultdict(self.default_value_func)
        self.value_range = default_value_range
        self.parent_key = None
        if data:
            self.update(data)
        
    def default_value_func(self):
        print("I am asked to give a default value.")
        return dict()

    def __setitem__(self, key, value): 
        self.storage[key] = value

    def __getitem__(self, key):
        try:
            return self.storage[key]
        except KeyError:
            v = self.default_value()
            self.storage[key] = v # store the newly generated default value
            return v

    def __delitem__(self, key):
        try:
            del self.storage[key]
        except KeyError:
            pass
    
    def __iter__(self):
        return iter(self.storage)
    
    def __len__(self):
        return len(self.storage)

    def max(self, key=None):
        return max(self.storage[key].values(), default=self.default_value())
    
    def min(self, key=None):
        return min(self.storage[key].values(), default=self.default_value())
    
    def argmax(self, key=None):
        return max(self.storage[key], key=self.storage[key].get, default=self.default_value())
    
    def argmin(self, key):
        return min(self.storage[key], key=self.storage[key].get, default=self.default_value())
    
    def default_value(self, key=None):
        return  (max(self.value_range) - min(self.value_range)) * np.random.random_sample() + min(self.value_range)

class StateActionFunction:
    def __init__(self, state_mgr=None, action_mgr=None, initial_value=None):
        if isinstance(state_mgr, grl.managers.StateManager):
            self.sm = state_mgr
        else:
            self.sm = grl.managers.StateManager()
        if isinstance(action_mgr, grl.managers.ActionManager):
            self.sm = action_mgr
        else:
            self.sm = grl.managers.ActionManager()
        self.init_value = initial_value
        self.v = dict()
        
    
    def update(self, s_l, a_l):
        pass

class HistoryActionFunction:
    def __init__(self):
        pass

class StateFunction:
    def __init__(self):
        pass

class ActionFunction:
    def __init__(self):
        pass

class HistoryFunction:
    def __init__(self):
        pass

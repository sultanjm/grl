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

    def __init__(self, dims=2, default=lambda: np.random.random_sample(), persist=False, data=None, root=None):
        self.dims = dims
        self.default = default
        self.storage = {}
        self.persist = persist
        self.root = root
        if data:
            self.update(data)

    def __setitem__(self, key, value): 
        self.storage[key] = value

    def __getitem__(self, key):
        try:
            return self.storage[key]
        except KeyError:
            if not self.root:
                self.root = self
            if self.dims > 1:
                v = Storage(dims=self.dims - 1, default=self.default, persist=self.persist, root=self.root)
                self.storage[key] = v # pretending that the storage is persistent
            else:
                v = self.default()
                if not self.persist:
                    # a non-existent value is accessed in a non-persistent storage
                    self.root.storage.clear()
                    # check for a memory leak!
                    del self.root
                else:
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

    def max(self):
        if self.dims == 1:
            return max(self.values(), default=self.default())
        else:
            return None
    
    def argmax(self):
        if self.dims == 1:
            # what is the default action when there is no element
            return max(self, key=self.get, default=self.default())
        else:
            return None

    def min(self):
        if self.dims == 1:
            return min(self.values(), default=self.default())
        else:
            return None
    
    def argmin(self):
        if self.dims == 1:
            return min(self, key=self.get, default=self.default())
        else:
            return None

    def __repr__(self):
        return dict.__repr__(self.storage)

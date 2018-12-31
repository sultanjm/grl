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

    def __init__(self, dims=2, default_val=(0,1), persist=True, default_arg=None, data=None, _root=None):
        self.dimensions = dims
        self.default_value = default_val
        self.storage = {}
        self.persist = persist

        # magic: (TODO remove this magic) if a deque is provided then it captures args
        if not isinstance(default_arg, collections.abc.Sequence):
            default_arg = collections.deque(maxlen=10)
        self.default_argument = default_arg

        self.root = _root

        if data:
            self.update(data)

    def __setitem__(self, key, value): 
        self.storage[key] = value
        # magic: source of the magic
        if isinstance(self.default_argument, collections.deque):
            self.default_argument.append(key)

    def __getitem__(self, key):
        try:
            return self.storage[key]
        except KeyError:
            if not self.root:
                self.root = self
            if self.dimensions > 1:
                v = Storage(dims=self.dimensions - 1, default_val=self.default_value, default_arg=self.default_argument, persist=self.persist, _root=self.root)
                self.storage[key] = v # pretending that the storage is persistent
            else:
                v = self.default_val()

                # magic: source of the magic
                if isinstance(self.default_argument, collections.deque):
                    self.default_argument.append(key)

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

    def default_val(self):
        return (max(self.default_value) - min(self.default_value)) * np.random.sample() + min(self.default_value)

    def default_arg(self):
        if not len(self.default_argument):
            raise RuntimeError("No default argument has been provided or captured yet.")
        arg = np.random.choice(len(self.default_argument))
        return self.default_argument[arg]

    def max(self):
        if self.dimensions == 1:
            return max(self.values(), default=self.default_val())
        else:
            return None
    
    def argmax(self):
        if self.dimensions == 1:
            return max(self, key=self.get, default=self.default_arg())
        else:
            return None

    def min(self):
        if self.dimensions == 1:
            return min(self.values(), default=self.default_value())
        else:
            return None
    
    def argmin(self):
        if self.dimensions == 1:
            return min(self, key=self.get, default=self.default_arg())
        else:
            return None

    def __repr__(self):
        return dict.__repr__(self.storage)

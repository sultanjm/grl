import grl
import collections
import numpy as np

class Storage(collections.MutableMapping):

    """ 
    Storage class requires default_arguments if argmax or argmin is needed.
    
    dimensions -- storage dimensions (default 2)
    default_values -- range of initial values as (min, max) (default (0,1))
    default_arguments -- list of default arguments (default None)
    persist -- persist the accessed-initialized variable (default True)
    compute_statistics -- enable statistics computation (default False)
    data -- set any initial data (default None)

    """

    def __init__(self, dimensions=2, data=None, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

        self.dimensions = dimensions
        self.storage = {}
        self.persist = self.kwargs.get('persist', True)

        self.default_values = self.kwargs.get('default_values', (0,1))
        self.default_arguments = self.kwargs.get('default_arguments', None)

        self.compute_statistics = self.kwargs.get('compute_statistics', False)
        if self.compute_statistics:
            self.max = max(self.default_values)
            self.min = min(self.default_values)
            self.argmax = None if not self.default_arguments else self.default_arguments[0]
            self.argmin = None if not self.default_arguments else self.default_arguments[-1]    

        if data:
            self.update(data)

        # internal parameters
        self.__root = self.kwargs.get('__root', None)

    def __setitem__(self, key, value):     
        self.storage[key] = value
        if self.dimensions == 1 and self.compute_statistics:
            self.update_statistics(key, value)

    def __getitem__(self, key):
        try:
            return self.storage[key]
        except KeyError:
            if not self.__root:
                self.__root = self
            if self.dimensions > 1:
                v = Storage(dimensions=self.dimensions - 1, default_values=self.default_values, default_arguments=self.default_arguments, persist=self.persist, compute_statistics=self.compute_statistics, __root=self.__root)
                # pretending that the storage is persistent
                self.storage[key] = v 
            else:
                v = self.default_val()

                if not self.persist:
                    # a non-existent value is accessed in a non-persistent storage
                    self.__root.storage.clear()
                    # check for a memory leak!
                    del self.__root
                else:
                    # store the newly generated default value
                    self.storage[key] = v
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

    def __repr__(self):
        return dict.__repr__(self.storage)

    def update_statistics(self, key, value):
        if self.max == self[self.argmax] and value > self.max: # an existant max value
            self.argmax, self.max = key, value

        if self.min == self[self.argmin] and value < self.min: # an existant min value
            self.argmin, self.min = key, value

    def default_val(self):
        return (max(self.default_values) - min(self.default_values)) * np.random.sample() + min(self.default_values)

    def default_arg(self):
        arg = None
        if self.default_arguments:
            arg = self.default_arguments[np.random.choice(len(self.default_arguments))]
        return arg

    def expectation(self, dist=None):
        if self.dimensions == 1:
            if not isinstance(dist, dict) or not np.isclose(sum(dist.values()), 1.0):
                keys = self.storage.keys()
                if keys:
                    dist = dict.fromkeys(keys, 1/len(keys))
                else:
                    return self.default_val()
            return sum([dist[key]*self.storage.get(key, self.default_val()) for key in dist.keys()])
        else:
            return None
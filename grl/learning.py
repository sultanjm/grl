import grl
import collections
import numpy as np

class Storage(collections.MutableMapping):

    """ 
    Important! Storage class requires default_arguments if max/min statistics is needed.
    
    dimensions -- storage dimensions (default 2)
    default_values -- range of initial values as (min, max) (default (0,1))
    default_arguments -- list of default arguments (default None)
    persist -- persist the access-initialized variable (default True)
    compute_statistics -- enable statistics computation (default False)
    data -- set any initial data (default None)

    """

    def __init__(self, dimensions=2, data=None, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

        # internal parameters
        self.root = self.kwargs.get('root', None)
        self.parent = self.kwargs.get('parent', None)
        self.key = self.kwargs.get('key', None)

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


    def __setitem__(self, key, value):     
        if self.dimensions == 1 and self.compute_statistics:
            self.update_statistics(key, value)
        self.storage[key] = value

    def __getitem__(self, key):
        try:
            v = self.storage[key]
            self.root = None
            return v
        except KeyError:
            # set the root carefully

            # parent 
            # if has siblings don't do anything
            # else:
            # tell parent to kill
            # p1 p2 p3.a {}
            # p1 p2 p3.b {v1, v2}

            if not self.root:
                self.root = self

            if self.dimensions > 1:
                v = Storage(dimensions=self.dimensions - 1, 
                            default_values=self.default_values, 
                            default_arguments=self.default_arguments, 
                            persist=self.persist, 
                            compute_statistics=self.compute_statistics, 
                            root=self.root,
                            parent=self,
                            key=key)
                self.storage[key] = v 
            else:
                v = self.default_val()
                self.storage[key] = v
                
                if not self.persist:
                    self.purge(key)
            return v

    def purge(self, child_key):
        if len(self.storage) == 1 and self.parent:
            self.parent.purge(self.key)
        else:
            del self.storage[child_key]

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

    # the current situation must be realizable
    # should not have a wrong max/min
    # problem: we only have set values
    # might have accessed values too (if persist)
    # [v1 v2 v3 . . .]
    # [v1]
    # case 1:
    # current value is max
    # do: set the max
    # else case 2:
    # current max is existant and different (invalid maximum)
    # do: FIND NEW MAX



    # shouldn't make a difference

    def update_statistics(self, key, value):
        if self.max != self[self.argmax] and self[self.argmax] == self[self.argmax]:
            # the current maximum is invalid, it has been updated.
            # do a traditional maximum search
            # such calls are very rare (check!)
            items = dict.fromkeys(self.default_arguments, max(self.default_values))
            items.update(self.storage)
            self.argmax = max(items, key=items.get)
            self.max = items[self.argmax]
            
        if self.min != self[self.argmin] and self[self.argmin] == self[self.argmin]:
            # the current minimum is invalid, it has been updated.
            # do a traditional minimum search
            # such calls are very rare (check!)
            items = dict.fromkeys(self.default_arguments, min(self.default_values))
            items.update(self.storage)
            self.argmin = min(items, key=items.get)
            self.min = items[self.argmin]

        if self.max < value:
            # the current value is the maximum
            self.argmax, self.max = key, value
        if self.min > value:
            # the current value is the minimum
            self.argmin, self.min = key, value

    def default_val(self):
        return (max(self.default_values) - min(self.default_values)) * np.random.sample() + min(self.default_values)

    def default_arg(self):
        return None if not self.default_arguments else self.default_arguments[np.random.choice(len(self.default_arguments))]

    def expectation(self, dist=None):
        if self.dimensions == 1:
            if not isinstance(dist, dict) or sum(dist.values()) != 1.0:
                keys = self.storage.keys()
                if keys:
                    dist = dict.fromkeys(keys, 1/len(keys))
                else:
                    return self.default_val()
            return sum([dist[key]*self.storage.get(key, self.default_val()) for key in dist.keys()])
        else:
            return None
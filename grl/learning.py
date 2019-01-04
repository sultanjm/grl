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
        self.parent = self.kwargs.get('parent', None)
        self.key = self.kwargs.get('key', None)

        self.dimensions = dimensions
        self.storage = {}
        self.persist = self.kwargs.get('persist', True)

        self.default_values = self.kwargs.get('default_values', (0,1))
        self.default_arguments = self.kwargs.get('default_arguments', None)
        
        if data:
            self.update(data)


    def __setitem__(self, key, value):    
        self.storage[key] = value
        
    def __getitem__(self, key):
        try:
            return self.storage[key]
        except KeyError:
            if self.dimensions > 1:
                v = Storage(dimensions=self.dimensions - 1, 
                            default_values=self.default_values, 
                            default_arguments=self.default_arguments, 
                            persist=self.persist, 
                            parent=self,
                            key=key)
                self.storage[key] = v 
            else:
                v = self.default_val()
                self.storage[key] = v
                # purge the non-existant branch
                if not self.persist:
                    self.purge(key)
            return v           

    def __delitem__(self, key):
        try:
            del self.storage[key]
        except KeyError:
            pass
    
    def __iter__(self):
        if self.dimensions == 1:
            missing_keys = set(self.default_arguments) - set(self.storage.keys())
            for k,v in self.storage.items():
                yield (v, k)
            for k in missing_keys:
                yield (self.default_val(), k)
        else:
            return iter(self.storage)

    def __len__(self):
        return len(self.storage)

    def __repr__(self):
        return dict.__repr__(self.storage)

    def purge(self, child_key):
        self.storage.pop(child_key, None)
        if not len(self.storage) and self.parent:
            self.parent.purge(self.key)
 
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

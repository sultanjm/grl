import grl
import collections
import numpy as np
import random
import copy

__all__ = ['Storage']

class Storage(collections.MutableMapping):

    """ 
    dimensions -- storage dimensions (default 2)
    default_values -- range of initial values as (min, max) (default (0,1))
    leaf_keys -- list of leaf keys (default ['<?>'])
    persist -- persist the access-initialized variable (default True)
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
        if not isinstance(self.default_values, collections.Sequence):
            self.default_values = [self.default_values]
        self.leaf_keys = self.kwargs.get('leaf_keys', None)
        self.missing_keys = set() if not self.leaf_keys else set(self.leaf_keys)

        if data:
            self.update(data)

    def set_leaf_keys(self, keys):
        self.leaf_keys = keys
        self.missing_keys = set() if not self.leaf_keys else set(self.leaf_keys)
    
    def __setitem__(self, key, value):    
        self.storage[key] = value
        if self.missing_keys: 
            self.missing_keys.discard(key)

    def __getitem__(self, key):
        try:
            return self.storage[key]
        except KeyError:
            if self.dimensions > 1:
                v = Storage(dimensions=self.dimensions - 1, 
                            default_values=self.default_values, 
                            leaf_keys=self.leaf_keys, 
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
                # storage is persistant, hance remove the key
                elif self.missing_keys:
                    self.missing_keys.discard(key)
            return v           

    def __delitem__(self, key):
        try:
            del self.storage[key]
        except KeyError:
            pass

    def __iter__(self):
        if self.dimensions == 1:
            for key in self.storage.keys():
                yield key
            missing = copy.deepcopy(self.missing_keys)
            for key in missing:
                yield key
        else:
            return iter(self.storage.keys())

    def __len__(self):
        return len(self.storage)

    def __repr__(self):
        return dict.__repr__(self.storage)

    def operate(self, other, operation):
        if self.dimensions != 1:
            raise RuntimeError("TODO: This operation is not supported at {}-dimensional storage yet!".format(self.dimensions))
        # TODO: find a way to get rid of these for-loops.
        mock = copy.deepcopy(self.storage)
        if not isinstance(other, collections.Mapping):
            for k in self.storage:
                mock[k] = operation(self[k], other)
            for k in self.missing_keys:
                mock[k] = operation(self[k], other)
        else:
            for k in self.storage:
                try:
                    mock[k] = operation(self[k], other[k])
                except KeyError:
                    pass
            for k in self.missing_keys:
                try:
                    mock[k] = operation(self[k], other[k])
                except KeyError:
                    pass
        return Storage(self.dimensions, mock, *self.args, **self.kwargs)

    def __add__(self, other):
        return self.operate(other, lambda x, y: x + y)
    __radd__ = __add__

    def __sub__(self, other):
        return self.operate(other, lambda x, y: x - y)
    __rsub__ = __sub__

    def __mul__(self, other):
        return self.operate(other, lambda x, y: x * y)
    __rmul__ = __mul__
    
    def __matmul__(self, other):
        return self.operate(other, lambda x, y: x @ y)
    __rmatmul__ = __matmul__
    
    def __mod__(self, other):
        return self.operate(other, lambda x, y: x % y)
    __rmod__ = __mod__

    def __truediv__(self, other):
        return self.operate(other, lambda x, y: x / y)
    __rtruediv__ = __truediv__

    def __floordiv__(self, other):
        return self.operate(other, lambda x, y: x // y)
    __rfloordiv__ = __floordiv__

    def __pow__(self, other):
        return self.operate(other, lambda x, y: x ** y)
    __rpow__ = __pow__

    def sum(self):
        if self.dimensions == 1:
            return sum(self.storage.values()) +  sum([self.default_val() for _ in range(len(self.missing_keys))])

    def max(self):
        if self.dimensions == 1:
            max_v = max(self.storage.values(), default=max(self.default_values))
            if len(self.missing_keys):
                if max_v < max(self.default_values):
                    max_v = max(self.default_values)
            return max_v
    
    def argmax(self):
        if self.dimensions == 1:
            try: 
                max_key = max(self, key=self.get)
            except ValueError: 
                return None
            if len(self.missing_keys):
                if self.storage[max_key] < max(self.default_values):
                    max_key = random.sample(self.missing_keys, 1)[0]
            return max_key

    def min(self):
        if self.dimensions == 1:
            min_v = min(self.storage.values(), default=min(self.default_values))
            if len(self.missing_keys):
                if min_v > min(self.default_values):
                    min_v = min(self.default_values)
            return min_v
    
    def argmin(self):
        if self.dimensions == 1:
            try: 
                min_key = max(self, key=self.get)
            except ValueError: 
                return None
            if len(self.missing_keys):
                if self.storage[min_key] > min(self.default_values):
                    min_key = random.sample(self.missing_keys, 1)[0]
            return min_key

    def purge(self, child_key):
        self.storage.pop(child_key, None)
        if not len(self.storage) and self.parent:
            self.parent.purge(self.key)
 
    def default_val(self):
        return random.uniform(min(self.default_values), max(self.default_values))

    def avg(self, p=None):
        if self.dimensions == 1:
            if isinstance(p, collections.Mapping):
                if self.leaf_keys:
                    q = dict.fromkeys(self.leaf_keys, 0)
                    q.update(p)
                    p = q
                return (self * p).sum()
            else:
                return self.sum() / (len(self) + len(self.missing_keys))

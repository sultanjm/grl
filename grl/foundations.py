import abc
import grl
import numpy as np
import copy

__all__ = ['GRLObject', 'Domain', 'Agent', 'BinaryMock']

class GRLObject(abc.ABC):

    def __init__(self, history_mgr=None, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        # a derived class must initialize the required managers
        if isinstance(history_mgr, grl.HistoryManager):
            self.hm = history_mgr
        else:
            self.hm = grl.HistoryManager()
        self.sm = grl.StateManager(self.transition_func)
        self.am = grl.ActionManager()
        self.pm = grl.PerceptManager(self.emission_func)
        self.rm = grl.RewardManager(self.reward_func)
        self.setup()
    
    def stats(self):
        raise NotImplementedError

    def reset(self):
        pass

    @abc.abstractmethod
    def setup(self): 
        raise NotImplementedError

    # default: same state transition loop
    def transition_func(self, s, a): return s

    # default: identity emission function
    def emission_func(self, s): return s

    # default: zero reward function
    def reward_func(self, a, e, h): return 0

class Domain(GRLObject):

    @abc.abstractmethod
    def react(self, action): 
        raise NotImplementedError

    @abc.abstractmethod
    def start(self):
        raise NotImplementedError

class Agent(GRLObject):
   
    # default: interacting with only one domain
    def interact(self, domain):
        if not isinstance(domain, Domain):
            raise RuntimeError("No valid domian is provided.")    
        self.am = domain.am # agent knows the available actions in the domain
        self.pm = domain.pm # agent knows the receivable percepts from the domain
        self.rm = domain.rm # agent knows the true reward function of the domain  

    @abc.abstractmethod
    def act(self, percept=None): 
        raise NotImplementedError

class BinaryMock(Domain):

    def setup(self):
        self.am.actions = [0, 1]
        self.r_dummy = self.kwargs.get('r_dummy', 0)
    
    def start(self):
        self.last_percept = self.domain.start()
        return self.last_percept

    def hook(self, domain):
        if not isinstance(domain, Domain):
            raise RuntimeError("No valid domian is provided.")    
        self.domain = domain
        # complete the action space
        act_len = len(self.domain.am.actions)
        self.d = int(np.ceil(np.log2(act_len)))
        diff = act_len - 2**self.d
        self.ext_actions = copy.deepcopy(self.domain.am.actions)
        for _ in range(diff):
            self.ext_actions.append(self.domain.am.actions[0])
        # new state space
        self.sm.states = list(range(self.d))
        self.sm.state = self.sm.states[0]
        # empty set of binary actions
        self.b = list()
        self.last_percept = None
    
    def react(self, b):
        self.b.append(b)
        self.sm.transit(b)
        if self.sm.state != 0:
            e = self.pm.perception(self.sm.state)
        else:
            e = self.domain.react(self.inv_binary_func(self.b))
            self.last_percept = e
            self.b.clear()
        return e

    def emission_func(self, s):
        return self.last_percept

    def reward_func(self, a, e, h):
        if self.sm.state:
            return self.r_dummy
        else:
            return self.domain.rm.r(a, e, self.domain.hm.history)

    def transition_func(self, s, b):
        return (s + 1) % self.d

    def binary_func(self, a):
        b = grl.int2bits(a)
        while len(b) < self.d: 
            b.append(0)
        return b

    def inv_binary_func(self, b):
        return self.ext_actions[grl.bits2int(b)]

    

import abc
import grl
import numpy as np
import copy
import math
import random 

__all__ = ['GRLObject', 'Domain', 'Agent', 'BinaryMock']

class GRLObject(abc.ABC):

    def __init__(self, history_mgr=None, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

        if isinstance(history_mgr, grl.HistoryManager):
            self.keep_history = False
            self.hm = history_mgr
        else:
            self.keep_history = True
            self.hm = grl.HistoryManager(maxlen=self.kwargs.get('max_history', None), state_map=self.state_map)
        self.sm = grl.StateManager(self.transition_func)
        self.am = grl.ActionManager()
        self.pm = grl.PerceptManager(self.emission_func)
        self.rm = grl.RewardManager(self.reward_func)
        self.setup()
    
    def stats(self):
        raise NotImplementedError

    def reset(self):
        return

    def setup(self): 
        return
    
    def on(self, event):
        return

    # default: state map to a '<?>' state
    def state_map(self, a, e, h, *args, **kwargs): return '<?>'

    # default: same state transition loop
    def transition_func(self, s, a): return s

    # default: identity emission function
    def emission_func(self, s): return s

    # default: zero reward function
    def reward_func(self, a, e, h): return 0

class Domain(GRLObject):

    @abc.abstractmethod
    def react(self, a, h):
        raise NotImplementedError

    @abc.abstractmethod
    def start(self, a=None):
        raise NotImplementedError

    def oracle(self, a, e, h, g, *args, **kwargs):
        raise NotImplementedError

class Agent(GRLObject):
   
    # default: interacting with only one domain
    def interact(self, domain):
        if not isinstance(domain, Domain):
            raise RuntimeError("No valid domian is provided.")
        self.am = domain.am # agent knows the available actions in the domain
        self.pm = domain.pm # agent knows the receivable percepts from the domain
        self.rm = domain.rm # agent knows the true reward function of the domain  

        # WARNING! Be mindful when using oracles. They are super powerful!
        self.oracle = domain.oracle # agent has access to the oracle of the domain
        

    @abc.abstractmethod
    def act(self, h):
        raise NotImplementedError

    @abc.abstractmethod
    def learn(self, a, e, h):
        raise NotImplementedError
    
    @abc.abstractmethod
    def start(self, e=None):
        raise NotImplementedError

class BinaryMock(Domain):

    def setup(self):
        self.am.actions = [0, 1]
        self.r_dummy = self.kwargs.get('r_dummy', 0.0)
        self.hm_ae = grl.HistoryManager(maxlen=self.hm.history.maxlen)
        self.domain = None
        self.restrict_A_cache = dict()

    def start(self, a=None):
        # Design Choice: Ignore the starting binary input and 
        # take a random action on the hooked domain
        a_org = None if a == None else random.sample(self.ext_actions, 1)[0]
        self.prev_e = self.domain.start(a_org)
        if a_org != None: self.hm_ae.history.append(a_org)
        self.hm_ae.history.append(self.prev_e)
        return self.prev_e

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
        self.prev_e = None
        self.restrict_A_cache.clear()

    def react(self, b, h=None):
        self.b.append(b)
        self.sm.transit(b)
        if self.sm.state != 0:
            e = self.pm.perception(self.sm.state)
        else:
            a = self.inv_binary_func(self.b)
            e = self.domain.react(a)
            self.hm_ae.history.append(a)
            self.hm_ae.history.append(e)
            self.prev_e = e
            self.b.clear()
        return e

    def oracle(self, b, e, h, g, *args, **kwargs):
        # TODO: the following logic can be optimized (cache the previous oracle calls)

        # temporarily pop the last ae-pair
        if len(self.hm_ae.history) > 2: # TODO: This is a hack!
            e_org = self.hm_ae.history.pop()
            a_org = self.hm_ae.history.pop()
        else:
            e_org, a_org = None, None

        g_org = g**self.d
 
        if e == None:
            q = self.domain.oracle(None, None, self.hm_ae.history, g_org, *args, **kwargs)
        else:
            q = self.domain.oracle(a_org, e_org, self.hm_ae.history, g_org, *args, **kwargs)

        # push back the temporarily popped ae-pair
        if a_org and e_org:
            self.hm_ae.history.append(a_org)
            self.hm_ae.history.append(e_org)

        q_bin = grl.Storage(1, default=0, leaf_keys=[0,1])
        # "wierd" masking of the unavailable actions
        # moves the action-values of the unavailable actions to -inf
        q_bin[0] = (q + self.restricted_action_space(self.b, 0)).max()
        q_bin[1] = (q + self.restricted_action_space(self.b, 1)).max()
        q_bin = g ** (self.d - self.sm.state - 1) * q_bin
        return q_bin

    def restricted_action_space(self, b_vector, b):
        b_key = ''.join(str(x) for x in b_vector) + str(b)
        if self.restrict_A_cache.get(b_key, None):
            return self.restrict_A_cache[b_key]
        # expensive computation!
        A = grl.Storage(1, default= -math.inf, leaf_keys=self.domain.am.actions)
        for a in self.ext_actions:
            bit_list = self.binary_func(a)
            for bits in bit_list:
                a_str = ''.join(str(x) for x in bits)
                if a_str.startswith(b_key): A[a] = 0.0
        # cache the computation
        self.restrict_A_cache[b_key] = A
        return A

    def emission_func(self, s):
        return self.prev_e

    def reward_func(self, a, e, h):
        if self.sm.state:
            return self.r_dummy
        else:
            # temporarily pop the last ae-pair
            if len(self.hm_ae.history) > 2: # TODO: This is a hack!
                e_org = self.hm_ae.history.pop()
                a_org = self.hm_ae.history.pop()
            else:
                e_org, a_org = None, None
            if e == None:
                r = self.domain.rm.r(None, None, self.hm_ae.history)
            else:
                r = self.domain.rm.r(a_org, e_org, self.hm_ae.history)
            # push back the temporarily popped ae-pair
            if a_org and e_org:
                self.hm_ae.history.append(a_org)
                self.hm_ae.history.append(e_org)
            return r

    def transition_func(self, s, b):
        return (s + 1) % self.d

    # this is not exactly a function (returns a list of bits)
    def binary_func(self, a):
        indices = [i for i, x in enumerate(self.ext_actions) if x == a]
        b_list = list()
        for i in indices:
            b = grl.int2bits(i)
            while len(b) < self.d: 
                b.append(0)
            b_list.append(b)
        return b_list

    def inv_binary_func(self, b):
        return self.ext_actions[grl.bits2int(b)]
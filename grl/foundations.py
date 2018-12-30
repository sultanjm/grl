import abc
import grl

class GRLObject(abc.ABC):

    def __init__(self, history_mgr=None, params=None):
        self.params = params
        # a derived class must initialize the required managers
        if isinstance(history_mgr, grl.managers.HistoryManager):
            self.hm = history_mgr
        else:
            self.hm = grl.managers.HistoryManager()
        # WARNING! This part of the code is not robust enough.
        # assuming 0 is a 'valid' labled state
        self.sm = grl.managers.StateManager(self.transition_func, 0, self.state_label_func, self.state_inv_label_func)
        # assuming 0 is a 'valid' labled action
        self.am = grl.managers.ActionManager([0], self.action_label_func, self.action_inv_label_func)
        self.pm = grl.managers.PerceptManager(self.emission_func, self.percept_label_func, self.percept_inv_label_func)
        self.rm = grl.managers.RewardManager(self, self.reward_func)
        self.setup()
    
    def stats(self):
        pass

    def reset(self):
        pass

    def setup(self):
        pass

    # default: same state transition loop
    def transition_func(self, s_l, a_l):
        return s_l

    # default: identity emission function
    def emission_func(self, s_l):
        return s_l

    # default: zero reward function
    def reward_func(self, s_l, a_l, s_nxt_l=None, e_nxt_l=None, h=None):
        return 0

    # default: identity state label function
    def state_label_func(self, s):
        return s
    
    # default: identity inverse state label function
    def state_inv_label_func(self, s_l):
        return s_l

    # default: identity percept label function
    def percept_label_func(self, e):
        return e
    
    # default: identity inverse percept label function
    def percept_inv_label_func(self, e_l):
        return e_l
    
    # default: identity action label function
    def action_label_func(self, a):
        return a

    # default: identity inverse action label function
    def action_inv_label_func(self, a_l):
        return a_l


class Domain(GRLObject):
        
    # default: a standard Partially Observed Markov Decision Problem (POMDP) implementation if the default emission_func is overloaded
    # otherwise, this is a standared Markov Decision Problem (MDP) react function
    @abc.abstractmethod
    def react(self, action_l):
        e_l = self.pm.l(self.pm.perception(self.sm.transit(self.am.inv_l(action_l))))
        self.hm.extendHistory(e_l, is_complete=True)
        return e_l

class Agent(GRLObject):
   
    # default: interacting with only one domain
    def interact(self, domain):
        if not isinstance(domain, Domain):
            raise RuntimeError("No valid domian is provided.")    
        self.am = domain.am # agent knows the available actions in the domain
        self.pm = domain.pm # agent knows the receivable percepts from the domain
        self.rm = domain.rm # agent knows the true reward function of the domain
        self.reset()  

    @abc.abstractmethod
    def act(self):
        pass 
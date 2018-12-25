import abc
import grl

class Domain(abc.ABC):

    # params - domain dependent parameters
    # history_mgr - a global history object of the framework

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
        self.pm = grl.managers.PerceptManager(self.emission_func, self.percept_label_func, self.percept_inv_label_func)
        self.actions = None
        self.setup()
        
    # default: a standard Partially Observed Markov Decision Problem (POMDP) implementation if the default emission_func is overloaded
    # otherwise, this is a standared Markov Decision Problem (MDP) react function
    def react(self, action):
        e_l = self.pm.l(self.pm.perception(self.sm.transit(action)))
        self.hm.extendHistory(e_l, is_complete=True)
        return e_l
    
    def stats(self):
        pass

    def reset(self):
        pass

    @abc.abstractmethod
    def setup(self):
        pass
    
    def setActions(self, actions):
        self.actions = actions

    def getActions(self):
        return self.actions

    # default: same state transition loop
    def transition_func(self, s_l, a):
        return s_l

    # default: identity emission function
    def emission_func(self, s_l):
        return s_l

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

class Agent(abc.ABC):

    # params - domain dependent parameters
    # hm_g - a global history object of the framework

    def __init__(self, actions=None, history_mgr=None, params=None):
        self.params = params
        self.actions = actions
        # a derived class must initialize the required managers
        if isinstance(history_mgr, grl.managers.HistoryManager):
            self.hm = history_mgr
        else:
            self.hm = grl.managers.HistoryManager()
        self.sm = grl.managers.StateManager(self.transition_func, 0, self.state_label_func, self.state_inv_label_func)
        self.setup()

    def setActions(self, actions):
        self.actions = actions
    
    def getActions(self):
        return self.actions

    # default: same state transition loop
    def transition_func(self, s_l, a):
        return s_l

    # default: identity state label function
    def state_label_func(self, s):
        return s
    
    # default: identity inverse state label function
    def state_inv_label_func(self, s_l):
        return s_l

    @abc.abstractmethod
    def act(self):
        pass

    def reset(self):
        pass

    def setup(self):
        pass
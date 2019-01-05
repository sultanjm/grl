import abc
import grl

__all__ = ['GRLObject', 'Domain', 'Agent']

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
        pass

    def reset(self):
        pass

    def setup(self):
        pass

    # default: same state transition loop
    def transition_func(self, s, a):
        return s

    # default: identity emission function
    def emission_func(self, s):
        return s

    # default: zero reward function
    def reward_func(self, a, e, h):
        return 0

class Domain(GRLObject):
        
    @abc.abstractmethod
    def react(self, action):
        pass


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
        pass 
    
    def learn(self, percept=None):
        pass
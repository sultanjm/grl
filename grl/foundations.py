import abc
import grl.managers

class Domain(abc.ABC):

    # params - domain dependent parameters
    # hm_g - a global history object of the framework

    def __init__(self, hm_g=None, params=None):
        self.params = params
        # a derived class must initialize the required managers
        if isinstance(hm_g, grl.managers.HistoryManager):
            self.hm = hm_g
        else:
            self.hm = grl.managers.HistoryManager()
        self.sm = grl.managers.StateManager()
        self.pm = grl.managers.PerceptManager()
        self.setup()

    @abc.abstractmethod
    def setup(self):
        pass

    @abc.abstractmethod
    def react(self, action):
        pass

    @abc.abstractmethod
    def reset(self):
        pass

class Agent(abc.ABC):

    # params - domain dependent parameters
    # hm_g - a global history object of the framework

    def __init__(self, actions, hm_g=None, params=None):
        self.params = params
        self.actions = actions
        # a derived class must initialize the required managers
        if isinstance(hm_g, grl.managers.HistoryManager):
            self.hm = hm_g
        else:
            self.hm = grl.managers.HistoryManager()
        self.sm = grl.managers.StateManager()
        self.setup()

    @abc.abstractmethod
    def setup(self):
        pass

    @abc.abstractmethod
    def act(self):
        pass

    @abc.abstractmethod
    def reset(self):
        pass
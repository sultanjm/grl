import grl
import numpy as np

class BlindMaze4x4(grl.foundations.Domain):

    # design philosophy:
    # should provide a standard setup
    # only replace/define the non-standard parts

    def setup(self):
        self.sm.setCurrentState(np.random.choice(16))
        self.setActions(np.arange(4))

    def transition_func(self,s,a):
        return np.random.choice(16)
    
    def reset(self):
        self.sm.setCurrentState(np.random.choice(16))


class RandomAgent(grl.foundations.Agent):

    def act(self):
        if  len(self.actions) < 1:
            raise RuntimeError("No set of actions is provided.")
        a = np.random.choice(self.actions)
        self.hm.extendHistory(a, is_complete=False)
        return a
    


history_mgr = grl.managers.HistoryManager(MAX_LENGTH=10)
domain = BlindMaze4x4(history_mgr)
agent = RandomAgent(domain.getActions(), history_mgr)

for t in range(15):
    domain.react(agent.act())

print(history_mgr.getHistory())
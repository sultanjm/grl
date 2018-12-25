import grl
import numpy as np

class BlindMaze4x4(grl.foundations.Domain):

    def setup(self):
        self.sm.setCurrentState(np.random.choice(16))
        self.sm.setTransitionMatrix(
            # TOOD: insert a 4x16x16 transition matrix here!
            grl.utilities.generateRandomProbabilityMatrix(4, 16, repeat_second_dimension=True)
        )
        self.pm.setEmissionMatrix(
            # TOOD: insert a 16x2 emission matrix here!
            grl.utilities.generateRandomProbabilityMatrix(16, 2, repeat_second_dimension=False)
        )
        #self.pm.setLabels([(0,0),(0,1)])

    def reset(self):
        self.sm.setCurrentState(np.random.choice(16))

    def react(self, action):
        self.sm.makeTransition(action)
        e = self.pm.getPercept(self.sm.s)
        self.hm.extendHistory(e, is_complete=True)
        return e


class RandomAgent(grl.foundations.Agent):

    def setup(self):
        pass

    def reset(self):
        pass

    def act(self):
        a = np.random.choice(self.actions)
        self.hm.extendHistory(a, is_complete=False)
        return a
    


hm = grl.managers.HistoryManager(MAX_LENGTH=10)
domain = BlindMaze4x4(hm)
agent = RandomAgent(np.arange(4), hm)
for t in range(15):
    a = agent.act()
    e = domain.react(a)
    print(a,e)

print(hm.getHistory())
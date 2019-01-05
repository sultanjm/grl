import grl

class RandomAgent(grl.Agent):
    def act(self, percept):
        return grl.epsilon_sample(self.am.actions)
import grl
import numpy as np

class BlindMaze4x4(grl.foundations.Domain):

    # design philosophy:
    # should provide a standard setup
    # only replace/define the non-standard parts

    def react(self, action_l):
        return super().react(action_l)

    def setup(self):
        self.sm.setCurrentState(np.random.choice(16))
        self.am.setActions(['up', 'down', 'left', 'right'])

    def transition_func(self, s_l, a_l):
        return np.random.choice(16)
    
    def reset(self):
        self.sm.setCurrentState(np.random.choice(16))

class BlindMaze8x8(grl.foundations.Domain):
    
    # design philosophy:
    # should provide a standard setup
    # only replace/define the non-standard parts
    
    def react(self, action_l):
        return super().react(action_l)

    def setup(self):
        self.sm.setCurrentState(np.random.choice(64))
        self.am.setActions(np.arange(8))

    def transition_func(self, s_l, a_l):
        return np.random.choice(16)
    
    def reset(self):
        self.sm.setCurrentState(np.random.choice(64))


class RandomAgent(grl.foundations.Agent):

    def act(self):
        a_l = np.random.choice(self.am.getActions())
        self.hm.extendHistory(a_l, is_complete=False)
        return a_l
    
class GreedyQAgent(grl.foundations.Agent):

    def setup(self):
        self.Q = grl.learning.Storage()
        self.alpha = grl.learning.Storage(default_value_range = 0.99)

    # def update(self, event):
    #     # event in the shape of (h,a,e,s)
    #     e = self.pm.inv_l(event.e_l)
    #     h = self.hm.getHistory()
    #     a = self.am.inv_l(event.a_l)
    #     s = self.sm.getCurrentState()
    #     nxt_s = self.map.state(history(h,a,e))

    #     self.Q[s] = self.Q.get((s,a), 0) + self.alpha.get((s,a), 0) * (self.Q.get((s,a), 0) - np.max(self.Q(nxt_s, :)))
    #     self.Q.update(s,a)


    #     self.Q()


        # should be able to perfrom EXPECTED, MAX and SOFTMAX on self.q
        # should be able to update the function
        # func.update()

    # def act(self):
    #     return self.am.l(np.argmax(self.Q[self.sm.s,:]))
    
    def learn(self):
        pass


##############################################################

# def phi(h):
#     return h

# history_mgr = grl.managers.HistoryManager(MAX_LENGTH=10, history_to_state_map=phi)
# domain_1 = BlindMaze4x4(history_mgr)
# domain_2 = BlindMaze8x8(history_mgr)
# agent = RandomAgent(history_mgr)

# for d in [domain_1, domain_2]:
#     agent.interact(d)
#     for t in range(15):
#         d.react(agent.act())

#     print(history_mgr.getHistory())

a = grl.learning.Storage()
a[1]
a[1][2]
a[2][2]
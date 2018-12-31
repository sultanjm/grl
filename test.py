import grl
import numpy as np

class BlindMaze4x4(grl.foundations.Domain):

    # design philosophy:
    # should provide a standard setup
    # only replace/define the non-standard parts

    def react(self, action):    
        e = self.pm.perception(self.sm.transit(action))
        self.hm.extend_history(e, complete=True)
        return e

    def setup(self):
        self.sm.state = np.random.choice(16))
        self.am.actions = frozenset(['u', 'd', 'l', 'r'])

    def transition_func(self, state, action):
        return np.random.choice(16)
    
    def reset(self):
        self.sm.state = np.random.choice(16)

class BlindMaze8x8(grl.foundations.Domain):
    
    # design philosophy:
    # should provide a standard setup
    # only replace/define the non-standard parts
    
    def react(self, a):    
        e = self.pm.perception(self.sm.transit(action))
        self.hm.extend_history(e, complete=True)
        return e

    def setup(self):
        self.sm.state = np.random.choice(64)
        self.am.actions = frozenset(['u', 'd', 'l', 'r', 'dl', 'dr', 'ul', 'ur'])

    def transition_func(self, s, a):
        return np.random.choice(16)
    
    def reset(self):
        self.sm.state = np.random.choice(64)


class RandomAgent(grl.foundations.Agent):

    def act(self, e=None):
        a = np.random.choice(self.am.actions)
        self.hm.extend_history(a, complete=False)
        return a
    
class GreedyQAgent(grl.foundations.Agent):

    def setup(self):
        self.Q = grl.learning.Storage(dims=2, default_val=1, persist=False) # optimistic initialization
        self.alpha = grl.learning.Storage(dims=2, default_val=0.99) # initial learning rate
        self.a = None
    # def update(self, event):
    #     # event in the shape of (h,a,e,s)
    #     e = self.pm.inv_l(event.e_l)
    #     h = self.hm.getHistory()
    #     a = self.am.inv_l(event.a_l)
    #     s = self.sm.getCurrentState()
    #     nxt_s = s #self.map.state(history(h,a,e))
    
    #     self.Q[s][a] = self.Q[s][a] + self.alpha[s][a] * (self.Q[s][a] - self.Q[nxt_s].max())
    #     self.alpha[s][a] = self.alpha[s][a] ** 2


    # #     self.Q()


    #     # should be able to perfrom EXPECTED, MAX and SOFTMAX on self.q
    #     # should be able to update the function
    #     # func.update()

    def act(self, percept_l):
        self.update(percept_l)
        return self.Q[percept_l].argmax()
    
    def update(self, e_l):
        self.Q[][a] = self.Q[s][a] + self.alpha[s][a] * (self.Q[s][a] - self.Q[nxt_s].max())
        self.alpha[s][a] = self.alpha[s][a] ** 2


##############################################################

# def phi(h):
#     return h

history_mgr = grl.managers.HistoryManager(MAX_LENGTH=10, history_to_state_map=phi)
domain = BlindMaze4x4(history_mgr)
agent = RandomAgent(history_mgr)
agent.interact(domain)

e = None
for t in range(15):
    e = domain.react(agent.act(e))

    print(history_mgr.getHistory())

# a = grl.learning.Storage(dims=3, persist=False)
# print(a[3][1][2] != a[3][1][2])
# print(a[2][1][1] != a[2][1][1])
# a[1][3][4] = 1.5
# print(a[1][3][4] == 1.5)
# print(a[1][3].max() == 1.5)
# print(a[1][3].argmax() == 4)
# print(len(a) == 1)

# b = grl.learning.Storage(persist=True)
# print(b[2][(1,2)] == b[2][(1,2)])
# b[1][3] = 2
# b[1][4] = 4
# print(b[1][3] == 2)
# print(b[1].expectation() == 3.0)
# print(b[1].expectation({3: 1.0, 4:0.0}) == 2.0)
# print(b[1].max() == 4)
# print(b[1].argmax() == 4)
# print(len(b) == 2)
# print(b[5].argmax())
# print(b[7].argmin())


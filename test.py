import grl
import numpy as np
import collections

class BlindMaze(grl.Domain):
    def react(self, action):    
        e = self.pm.perception(self.sm.transit(action))
        return e
        
    def setup(self):
        self.maze_len = self.kwargs.get('maze_len', 4)
        self.sm.states = [(x,y) for x in range(self.maze_len) for y in range(self.maze_len)]
        self.am.actions = ['u', 'd', 'l', 'r']
        self.sm.state = self.sm.states[np.random.choice(len(self.sm.states))]

    def transition_func(self, state, action):
        if action == 'u':
            s_nxt = (state[0], min(state[1] + 1, self.maze_len - 1))
        elif action == 'd':
            s_nxt = (state[0], max(state[1] - 1, 0))
        elif action == 'l':
            s_nxt = (max(state[0] - 1, 0), state[1])
        elif action == 'r':
            s_nxt = (min(state[0] + 1, self.maze_len - 1), state[1])
        else:
            s_nxt = state
        return s_nxt
    
    def emission_func(self, state):
        if state == (0,0):
            e = ('o_o', 1)
            #e = (state, 1)
            self.reset()
        else:
            e = ('-_-', 0)
            #e = (state, 0)
        return e
    
    def reward_func(self, a, e, h):
        return e[1]

    def reset(self):
        self.sm.state = self.sm.states[np.random.choice(len(self.sm.states))]


class SimpleMDP(grl.Domain):
    def react(self, action):    
        e = self.pm.perception(self.sm.transit(action))
        return e

    def setup(self):
        self.sm.states = ['s-left', 's-right']
        self.am.actions = ['left', 'right']
        self.sm.state = self.sm.states[np.random.choice(len(self.sm.states))]

    def transition_func(self, state, action):
        s_nxt = state
        if state == 's-left' and action == 'left':
            s_nxt = 's-right'
        elif state == 's-right' and action == 'right':
            s_nxt = 's-left'
        return s_nxt
    
    def reward_func(self, a, e, h):
        s = self.sm.prev_state
        if (s == 's-left' and a == 'left') or (s == 's-right' and a == 'right'):
            return 1
        else:
            return 0

    def reset(self):
        self.sm.state = self.sm.states[np.random.choice(len(self.sm.states))]

class RandomAgent(grl.Agent):
    def act(self, percept):
        return grl.epsilon_sample(self.am.actions)

class GreedyQAgent(grl.Agent):
    def setup(self):
        self.epsilon = self.kwargs.get('exploration_factor', 0.1)
        self.g = self.kwargs.get('discount_factor', 0.999)
        self.Q = grl.Storage(dimensions=2, default_values=self.kwargs.get('Q_init', (0,1)), persist=self.kwargs.get('Q_persist', False), compute_statistics=True)
        self.alpha = grl.Storage(dimensions=2, default_values=self.kwargs.get('learning_rate_init', (0.999, 0.999)))
    
    def interact(self, domain):
        super().interact(domain)
        self.Q.set_default_arguments(self.am.actions)

    def act(self, e):

        a = self.am.action

        # This is not the first percept received from the domain.
        # This part looks ugly.
        if a:
            s = self.hm.mapped_state()
            s_nxt = self.hm.mapped_state(a, e)
            r_nxt = self.rm.r(a, e, self.hm.history)

            self.Q[s][a] = self.Q[s][a] + self.alpha[s][a] * (1-self.g) * (r_nxt + self.g * max(self.Q[s_nxt])[0] - self.Q[s][a])
            self.alpha[s][a] = self.alpha[s][a] * 0.999
        else:
            s = e

        self.am.action = grl.epsilon_sample(self.am.actions, max(self.Q[s])[1], 0.1)

        return self.am.action


################################################################

# scenario 1:
# A <- dummy_e
# A -> a

# need to define these two

# A(a|empty) ~ A_0(a)
# E(e|empty,a)

# The agent can not do better than a random agent in the start, unless it has some prior knowledge/model
# of the domain.
# One the otherhand, humans do not act first. But in this situation, the world is not a purely reactive
# simulator, or is it? The act of "doing nothing" is a valid action in the real-world.

# scenario 2:
# E <- dummy_a
# E -> e

# E(e|empty, empty) ~ E_0(e)
# A(a|e)

# The scenario 2 is more typical where there is an initial distributions on the percepts 
# (or intern on states.)


##############################################################

# d = grl.learning.Storage(dimensions=1, persist=True, default_arguments=['a', 'b', 'c'])
# d['a'] = 4
# d['c']
# print(max(d)[1])
# print(min(d)[0])

# for k in d:
#     print(k)
# print(len(d))
# d['b'] = 3
# for k in d:
#     print(k)
     
# print(len(d))

def phi_percept(a, e, h):
    # extract last percept
    if e:
        return e
    else:
        return h[-1][-1]

history_mgr = grl.HistoryManager(maxlen=10, state_map=phi_percept)
#domain = BlindMaze(history_mgr, maze_len=2)
domain = SimpleMDP(history_mgr)
#agent = RandomAgent(history_mgr)
agent = GreedyQAgent(history_mgr, Q_persist=False)

a_nxt = None 
# The iteration starts from the domain, which is more natural. The domain
# can set the initial state (if any) that consequently sets the initial percept.
agent.interact(domain)

for t in range(100000):
    a = a_nxt
    e = domain.react(a)
    a_nxt = agent.act(e)
    history_mgr.extend_history((a,e))

print('Action-value Function:')
print(agent.Q)
print('Optimal Policy:')
print(grl.optimal_policy(agent.Q))
#print('Learning Rates:')
#print(agent.alpha)
#print(history_mgr.history)

# a = grl.learning.Storage(dimensions=3, persist=False)
# a[1][2][3]=4 # {1: {2: {3: 4}}}
# a[0][2][3]
# #a[1][2]=4 # {1: {2: 4}}

# print(a)
# print(a[3][1][2] != a[3][1][2])
# print(a[2][1][1] != a[2][1][1])
# a[1][3][4] = 1.5
# print(a[1][3][4] == 1.5)
# print(a[1][3].max == 1.5)
# print(a[1][3].argmax == 4)
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


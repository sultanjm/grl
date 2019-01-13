import grl
import numpy as np
import collections
from examples import *


def phi_percept(a, e, h):
    # extract last percept
    if e is not None:
        return e
    else:
        return h[-1]

history_mgr = grl.HistoryManager(maxlen=10, state_map=phi_percept)
#domain = BlindMaze(history_mgr, maze_len=2)
domain = SimpleMDP(history_mgr)
#agent = RandomAgent(history_mgr)
agent = FrequencyAgent(history_mgr)
#agent = GreedyQAgent(history_mgr, value_function_persist=False, exploration_factor=0.3)
#bin_domain = grl.BinaryMock(history_mgr)

#bin_domain.hook(domain)
agent.interact(domain)

h = history_mgr.history

# A Domain-Initiated Framework
# h.append(bin_domain.start())

# An Agent-Initiated Framework
a = agent.start()
e = domain.start(a)
h.append(a)
h.append(e)

for t in range(100):
    a = agent.act(h)
    e = domain.react(a, h)
    agent.learn(a, e, h)
    h.append(a)
    h.append(e)

print(agent.v)
print(agent.p)
print(agent.r)
print(agent.pi)

# print('Action-value Function:')
# print(agent.Q)
# print('Optimal Policy:')
# print(grl.optimal_policy(agent.Q))
# #print('Learning Rates:')
# #print(agent.alpha)
# print(history_mgr.history)


# a = grl.Storage(2, leaf_keys=[1,2,3,4], default_values=0.5, persist=False)
# b = grl.Storage(2, leaf_keys=[1,2,3,4,5], default_values=0)

# a[1][1] = 1
# a[1][2] = 2
# b[2][1] = 1
# b[2][2] = 2
# a[1] = {1: 2, 2: 4, 4: 5} + a[1]
# print(a[1])
# print(a[1].max())
# print(a[1].argmax())
# print(a[1].min())
# print(a[1].argmin())
# print(a[1].sum())
# print(a[1].avg())
# print(a[3].max())
# print(a[3].argmax())
# print(a[3].min())
# print(a[3].argmin())
# print(a[3].sum())
# print(a[3].avg())
# print(b[3].argmin())

# print(a[1].avg())
# print(a[1].avg({1:0, 2:0, 3:5.0}))
# print(a[1].sum())

# p = grl.Storage(3, default=0, leaf_keys=[1,2])
# r = grl.Storage(3, default=0, leaf_keys=[1,2])
# pi = grl.Storage(2, default=0, leaf_keys=['a', 'b'])
# p[1]['a'][1] = 1
# p[1]['b'][2] = 1
# p[2]['a'][1] = 1
# p[2]['b'][2] = 1

# r[1]['a'][1] = 1
# v = ValueIteration(p, r)
# print(v)
# pi[1]['a'] = 0.3
# pi[1]['b'] = 0.7
# pi[2]['a'] = 0.5
# pi[2]['b'] = 0.5
# print(PolicyImprovement(p, r, v, pi))
import grl
import numpy as np
import collections
from examples import *


def phi_last_percept(h, *args, **kwargs):
    # extract last percept
    return h[-1]

def phi_extreme_va(h, *args, **kwargs):
    eps = kwargs.get('eps', 0.01)
    q_func = kwargs.get('q_func', None)
    q = q_func(h, *args, **kwargs)
    s = (q.max() // eps, q.argmax())
    return s

def phi_extreme_q(h, *args, **kwargs):
    eps = kwargs.get('eps', 0.01)
    q_func = kwargs.get('q_func', None)
    q = q_func(h, *args, **kwargs)
    q = q // eps
    s = tuple(q[k] for k in sorted(q))
    return s

history_mgr = grl.HistoryManager(maxlen=10, state_map=phi_extreme_va)
#domain = BlindMaze(history_mgr, maze_len=2)
domain = SimpleMDP(history_mgr)
#agent = RandomAgent(history_mgr)
agent = FrequencyAgent(history_mgr)
#agent = GreedyQAgent(history_mgr, value_function_persist=False, exploration_factor=0.3)

o_domain = domain
domain = grl.BinaryMock(history_mgr)
domain.hook(o_domain)

agent.interact(domain)

# A Domain-Initiated Framework
# e = domain.start()
# history_mgr.record([e])

# An Agent-Initiated Framework
a = agent.start()
e = domain.start(a)
history_mgr.record([a,e])

h = history_mgr.h
for t in range(1000):
    a = agent.act(h)
    e = domain.react(a, h)
    agent.learn(a, e, h)
    history_mgr.record([a,e])

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
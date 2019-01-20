import grl
import numpy as np
import collections
from examples import *


p = grl.Storage(3, leaf_keys=[0,1])
p[1]['a'][0] = 1
p[1]['a'][1] = 1
p[1]['b'][0] = 1
p[1]['b'][1] = 1
p[2]['a'][0] = 1
p[2]['a'][1] = 1
p[2]['b'][0] = 1
p[2]['b'][1] = 1
assert(p[1]['a'].sum() == 2)
assert(p[1]['b'].sum() == 2)
assert(p.sum() == 8)
assert(p[1].sum() == 4)
assert(p[2].sum() == 4)
print(p.sum())
# hm = grl.HistoryManager()
# hm.record(['a',1,'b',2])
# assert(hm.h.extract(1) == 'b')
# assert(hm.h.extract(2) == 2)
# hm.record(['c',3,'d',4])
# assert(hm.h.extract(1, grl.Index.PREVIOUS) == 'c')
# assert(hm.h.extract(2, grl.Index.PREVIOUS) == 3)
# hm.extend('e')
# assert(hm.h.extract(1) == 'e')
# assert(hm.h.extract(2) == 4)
# assert(hm.h.extract(1, grl.Index.PREVIOUS) == 'd')
# assert(hm.h.extract(2, grl.Index.PREVIOUS) == 3)
# hm.extend([5])

def phi_extreme_a(h, *args, **kwargs):
    q_func = kwargs.get('q_func', None)
    q = q_func(h, *args, **kwargs)
    s = q.argmax()
    return s

def phi_extreme_va(h, *args, **kwargs):
    eps = kwargs.get('eps', 0.1)
    q_func = kwargs.get('q_func', None)
    q = q_func(h, *args, **kwargs)
    s = (q.max() // eps, q.argmax())
    return s

def phi_extreme_q(h, *args, **kwargs):
    eps = kwargs.get('eps', 0.1)
    q_func = kwargs.get('q_func', None)
    q = q_func(h, *args, **kwargs)
    q = q // eps
    s = tuple(q[k] for k in sorted(q))
    return s

def phi_last_percept(h, *args, **kwargs):
    # extract last percept
    return h[-1]


history_mgr = grl.HistoryManager(maxlen=10, state_map=phi_extreme_q)
#domain = BlindMaze(history_mgr, maze_len=2)
#domain = SimpleMDP(history_mgr)
domain = DynamicKeys(history_mgr)
#agent = RandomAgent(history_mgr)
agent = FrequencyAgent(history_mgr)
#agent = GreedyQAgent(history_mgr, value_function_persist=False, exploration_factor=0.3)

# o_domain = domain
# domain = grl.BinaryMock(history_mgr)
# domain.hook(o_domain)

agent.interact(domain)

history_mgr.register(domain, grl.EventType.ADD)
history_mgr.register(domain, grl.EventType.REMOVE)

# A Domain-Initiated Framework
# e = domain.start()
# history_mgr.record([e])

# An Agent-Initiated Framework
a = agent.start()
e = domain.start(a)
history_mgr.record([a,e])

h = history_mgr.h
for t in range(10000):
    a = agent.act(h)
    e = domain.react(h, a)
    agent.learn(h, a, e)
    history_mgr.record([a,e])

print(agent.v)
print(agent.p)
print(agent.r)
print(agent.pi)

print(agent.stats())

# print('Action-value Function:')
# print(agent.Q)
# print('Optimal Policy:')
# print(grl.optimal_policy(agent.Q))
# #print('Learning Rates:')
# #print(agent.alpha)
# print(history_mgr.history)
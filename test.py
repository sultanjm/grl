import grl
import numpy as np
import collections
from examples import *
import time
import datetime

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

def phi_extreme_nq(h, *args, **kwargs):
    eps = kwargs.get('eps', 0.0001)
    q_func = kwargs.get('q_func', None)
    q = q_func(h, *args, **kwargs)
    q = (q / q.max()) // eps
    s = tuple(q[k] for k in sorted(q))
    return s

def phi_last_percept(h, *args, **kwargs):
    # extract last percept
    return h[-1]


history_mgr = grl.HistoryManager(maxlen=10, state_map=phi_extreme_nq)
#domain = BlindMaze(history_mgr, maze_len=2)
#domain = SimpleMDP(history_mgr)
domain = SlipperyHill(history_mgr)
#domain = DynamicKeys(history_mgr)
#agent = RandomAgent(history_mgr)
agent = FrequencyAgent(history_mgr, exploration_factor=0.1, discount_factor=0.999)
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
T = 1000
running_time = 0.0
t0 = time.time()
for t in range(T + 1):
    a = agent.act(h)
    e = domain.react(h, a)
    agent.learn(h, a, e)
    history_mgr.record([a,e])
    if t/T*100000 % 1 == 0.0:
        t1 = time.time()
        running_time += t1-t0
        print("\r{:.2f}% Complete - Elapsed: {}, Remaining: {}".format(t/T*100, datetime.timedelta(seconds=running_time), datetime.timedelta(seconds=running_time*(T-t)/(t+1))), end='')
        t0 = time.time()
print('\n')

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
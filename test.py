import grl
import numpy as np
import collections
from examples import *

def phi_percept(a, e_nxt, h):
    # extract last percept
    if e_nxt is not None:
        return e_nxt
    else:
        return h[-1]

history_mgr = grl.HistoryManager(maxlen=10, state_map=phi_percept)
#domain = BlindMaze(history_mgr, maze_len=2)
domain = SimpleMDP(history_mgr)
#agent = RandomAgent(history_mgr)
agent = GreedyQAgent(history_mgr, value_function_persist=False, exploration_factor=1.0)
bin_domain = grl.BinaryMock(history_mgr)

# basically, a binarizer should be a wrapper around the domain

bin_domain.hook(domain)
agent.interact(bin_domain)

h = history_mgr.history


#A(a0) -|
#       ---> two inits
#P(e0) -|

#A(a1|a0e0)
#P(e1|e0a0)


#A(a1)
#P(e2|a1) <- needs previous state
#A(a3|a1e2)
#P(e4|a1e2a3)
#A(a5|a1e2a3e4)
#...

#P(e1)
#A(a2|e1) <- needs previous action
#P(e3|e1a2)
#A(a4|e1a2e3)
#P(e5|e1a2e3a4)
#...


# An Agent-Initiated Framework
a = agent.start()
e = bin_domain.start(action=a)
h.append(a).append(e)

for t in range(1000):
    a = agent.act(h)
    e = bin_domain.react(h,a)
    agent.learn(h, a, e)
    h.append(a).append(e)


# A Domain-Initiated Framework
h.append(bin_domain.start())

for t in range(1000):
    a = agent.act(h)
    e = bin_domain.react(h,a)
    agent.learn(h, a, e)
    h.append(a).append(e)

print('Action-value Function:')
print(agent.Q)
print('Optimal Policy:')
print(grl.optimal_policy(agent.Q))
#print('Learning Rates:')
#print(agent.alpha)
print(history_mgr.history)

# The iteration starts from the domain, which is more natural. The domain
# can set the initial state (if any) that consequently sets the initial percept.
# agent.interact(domain)

# a_nxt = None 
# for t in range(100):
#     a = a_nxt
#     e = domain.react(a)
#     a_nxt = agent.act(e)
#     history_mgr.extend_history((a,e))

# print('Action-value Function:')
# print(agent.Q)
# print('Optimal Policy:')
# print(grl.optimal_policy(agent.Q))
#print('Learning Rates:')
#print(agent.alpha)
#print(history_mgr.history)

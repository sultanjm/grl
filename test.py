import grl
import numpy as np
import collections
from examples import *

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

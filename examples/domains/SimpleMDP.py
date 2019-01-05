import grl
import numpy as np

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

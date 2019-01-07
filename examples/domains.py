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

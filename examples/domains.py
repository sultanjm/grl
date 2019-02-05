import grl
import numpy as np

class SimpleMDP(grl.Domain):
    def react(self, h, a):
        e = self.pm.perception(self.sm.transit(a))
        return e

    def start(self, a=None, order=2):
        self.order = order
        return self.pm.perception(self.sm.state)

    def setup(self):
        self.sm.state_space = ['s-left', 's-right']
        self.am.action_space = ['left', 'right']
        self.sm.state = grl.epsilon_sample(self.sm.state_space)

    def transition_func(self, s, a):
        s_next = s
        if s == 's-left' and a == 'left':
            s_next = 's-right'
        elif s == 's-right' and a == 'right':
            s_next = 's-left'
        return s_next
    
    def reward_func(self, h):
        # TODO: only synced iterations
        assert(h.t - self.sm.hm.h.t == 1.0)
        agent_order = (self.order - 1) if (self.order - 1) else h.steplen
        a = h.extract(agent_order)
        s = self.sm.hm.h[-1]
        if (s == 's-left' and a == 'left') or (s == 's-right' and a == 'right'):
            return 1
        else:
            return 0

    def reset(self):
        self.sm.state = grl.epsilon_sample(self.sm.state_space)
    
    def oracle(self, h, *args, **kwargs):
        g = kwargs.get('g', 0.999)
        Q = grl.Storage(1, default=0, leaf_keys=self.am.action_space)
        s = h.extract(self.order)
        
        if s == 's-left':
            Q['left'] = 1/(1-g)
        elif s == 's-right':
            Q['right'] = 1/(1-g)
    
        return Q

    def on(self, event):
        grl.occurrence_ratio_processor(type(self).__name__, 's-left', event)

class SlipperyHill(grl.Domain):
    def setup(self):
        self.sm.state_space = [0, 1]
        self.pm.percept_space = ['@', '#']
        self.am.action_space = ['up', 'stay', 'down']
        self.sm.state = self.sm.state_space[0]
        self.optimal_actions = {0:'up', 1:'stay'}
        self.theta = 0.999
        self.C0 = 0.0001
        self.C1 = 1.0
        self.pmin = 0.001

    def start(self, a=None, order=2):
        self.order = order
        return '@'
        
    def react(self, h, a):
        bang_ratio = h.stats.get(type(self).__name__, dict()).get('#', self.pmin)
        self.sm.hm.record([self.sm.state])
        if a == self.optimal_actions[self.sm.state]:
            e_next = grl.epsilon_sample(self.pm.percept_space, '#', 1 - bang_ratio)
            if (e_next == '#' and self.sm.state == 0) or (e_next == '@' and self.sm.state == 1):
                self.sm.state = (self.sm.state + 1) % len(self.sm.state_space)
        else:
            self.sm.state = 0
            e_next = '@'
        return e_next

    def reset(self):
        self.sm.state = self.sm.state_space[0]
    
    def on(self, event):
        grl.occurrence_ratio_processor(type(self).__name__, '#', event, self.pmin)

    def reward_func(self, h):
        assert(h.t - self.sm.hm.h.t == 1.0)
        agent_order = (self.order - 1) if (self.order - 1) else h.steplen
        p_h = h.stats.get(type(self).__name__, dict()).get('#', self.pmin)
        a = h.extract(agent_order)
        s = self.sm.hm.h[-1]
        s_next = self.sm.state
        odd_factor = self.theta*(self.C1-self.C0)/(1-self.theta)
        normalizer = (2-self.pmin)*odd_factor + 1/self.pmin #1
        r_f_0 = odd_factor - p_h * odd_factor #0 - p_h * odd_factor
        r_f_1 = 2*odd_factor - p_h * odd_factor #odd_factor - p_h * odd_factor

        if s == 0 and a != self.optimal_actions[0]:
            return r_f_0/normalizer
        if s == 1 and a != self.optimal_actions[1]:
            return r_f_1/normalizer
        
        r_optimal = {
            (0, self.optimal_actions[0], 0): r_f_0/normalizer,
            (0, self.optimal_actions[0], 1): (r_f_0 + (self.C0/p_h))/normalizer,
            (1, self.optimal_actions[1], 0): r_f_1/normalizer,
            (1, self.optimal_actions[1], 1): (r_f_1 + (self.C1/p_h))/normalizer,
        }
        return r_optimal.get((s,a,s_next), 0)
    
    def oracle(self, h, *args, **kwargs):
        Q = grl.Storage(1, default=self.theta/(1-self.theta), leaf_keys=self.am.action_space)
        if self.sm.hm.h.t == h.t:
            s = self.sm.hm.h[-1]
        elif h.t - self.sm.hm.h.t == 1.0:
            s = self.sm.state
        if s == 0:
            Q *= self.C0
            Q[self.optimal_actions[0]] /= self.theta
        else:
            Q[self.optimal_actions[1]] /= self.theta
        return Q

class DynamicKeys(grl.Domain):

    def setup(self):
        self.sm.state_space = [0, 1, 2, 3]
        self.pm.percept_space = [':)', ':(']
        self.am.action_space = ['x', 'y', 'z']
        self.sm.state = self.sm.state_space[0]
        self.optimal_actions = {0:'x', 1:'y', 2:'x', 3:'y'}

    def start(self, a=None, order=2):
        self.order = order
        return ':('
        
    def react(self, h, a):
        happy_ratio = h.stats.get(type(self).__name__, dict()).get(':)', 0.0)
        self.sm.hm.record([self.sm.state])
        if a == self.optimal_actions[self.sm.state]:
            e_next = grl.epsilon_sample(self.pm.percept_space, ':)', 1 - happy_ratio)
            if e_next == ':)':
                self.sm.state = (self.sm.state + 1) % len(self.sm.state_space)
        else:
            e_next = ':('
        return e_next

    def reset(self):
        self.sm.state = self.sm.state_space[0]
    
    def on(self, event):
        grl.occurrence_ratio_processor(type(self).__name__, ':)', event)

    def reward_func(self, h):
        assert(h.t - self.sm.hm.h.t == 1.0)
        agent_order = (self.order - 1) if (self.order - 1) else h.steplen
        a = h.extract(agent_order)
        s = self.sm.hm.h[-1]

        if a == self.optimal_actions[s]:
            return 1
        else:
            return 0
    
    def oracle(self, h, *args, **kwargs):
        Q = grl.Storage(1, default=h.stats.get(type(self).__name__, dict()).get(':)', 0.0), leaf_keys=self.am.action_space)
        if self.sm.hm.h.t == h.t:
            Q[self.optimal_actions[self.sm.hm.h[-1]]] += 1.0
        elif h.t - self.sm.hm.h.t == 1.0:
            Q[self.optimal_actions[self.sm.state]] += 1.0
        return Q

class BlindMaze(grl.Domain):
    def react(self, h, a):
        e = self.pm.perception(self.sm.transit(a))
        return e
        
    def start(self, a=None, order=2):
        self.order = order
        return self.pm.perception(self.sm.state)

    def setup(self):
        self.maze_len = self.kwargs.get('maze_len', 4)
        self.sm.state_space = [(x,y) for x in range(self.maze_len) for y in range(self.maze_len)]
        self.am.action_space = ['u', 'd', 'l', 'r']
        self.sm.state = random.sample(self.sm.state_space, 1)[0]

    def transition_func(self, s, a):
        if a == 'u':
            s_next = (s[0], min(s[1] + 1, self.maze_len - 1))
        elif a == 'd':
            s_next = (s[0], max(s[1] - 1, 0))
        elif a == 'l':
            s_next = (max(s[0] - 1, 0), s[1])
        elif a == 'r':
            s_next = (min(s[0] + 1, self.maze_len - 1), s[1])
        else:
            s_next = s
        return s_next
    
    def emission_func(self, s):
        if s == (0,0):
            #e = ('o_o', 1)
            e = (s, 1)
            self.reset()
        else:
            #e = ('-_-', 0)
            e = (s, 0)
        return e
    
    def reward_func(self, h):
        return h[-1][1]

    def reset(self):
        self.sm.state = random.sample(self.sm.state_space, 1)[0]

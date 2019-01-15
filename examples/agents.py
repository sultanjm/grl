import grl
import numpy as np
import math

class RandomAgent(grl.Agent):

    def act(self, h):
        return grl.epsilon_sample(self.am.actions)

    def learn(self, a, e, h):
        pass

class GreedyQAgent(grl.Agent):
    def setup(self):
        self.epsilon = self.kwargs.get('exploration_factor', 0.1)
        self.g = self.kwargs.get('discount_factor', 0.999)
        self.Q = grl.Storage(dimensions=2, 
                             default_range=self.kwargs.get('value_function_init', (0,1)), 
                             persist=self.kwargs.get('value_function_persist', False))
        self.alpha = grl.Storage(dimensions=2, 
                                 default_values=self.kwargs.get('learning_rate_init', 0.999))

    def interact(self, domain):
        super().interact(domain)
        self.Q.set_leaf_keys(self.am.actions)

    def start(self, e = None):
        self.am.action = grl.epsilon_sample(self.am.actions)
        return self.am.action

    def act(self, h):
        s = self.hm.mapped_state()
        self.am.action = grl.epsilon_sample(self.am.actions, max(self.Q[s])[1], 0.1)
        return self.am.action
    
    def learn(self, a, e, h):
        s = self.hm.mapped_state()
        s_nxt = self.hm.mapped_state(a, e)
        r_nxt = self.rm.r(a, e, self.hm.history)
        
        self.Q[s][a] = self.Q[s][a] + self.alpha[s][a] * (r_nxt + self.g * max(self.Q[s_nxt])[0] - self.Q[s][a])
        self.alpha[s][a] = self.alpha[s][a] * 0.999

        if self.keep_history: self.hm.history.append(a).append(e)

class InternalStateAgent(grl.Agent):
    pass

class FrequencyAgent(grl.Agent):
    def setup(self):
        self.n = grl.Storage(3, default=0)
        self.r = grl.Storage(3, default=0)
        self.p = grl.Storage(3, default=0)
        self.g = self.kwargs.get('discount_factor', 0.999)
        self.eps = self.kwargs.get('tolerance', 1e-6)
        self.steps = self.kwargs.get('steps', math.inf)
        self.v = grl.Storage(1, default=0)
        self.pi = grl.Storage(2, default=1)

    def interact(self, domain):
        super().interact(domain)
        self.pi.set_leaf_keys(self.am.actions)
        self.pi.set_default(1/len(self.am.actions))

    def act(self, h):
        self.pi, self.v = grl.PITabular(self.p, self.r, self.v, self.pi, g=self.g, steps=1, vi_steps=1)
        # Oracle Alert!
        s = self.hm.state_map(None, None, h, self.g, q_func=self.oracle)
        return grl.epsilon_sample(self.am.actions, self.pi[s].argmax(), 1.0)

    def learn(self, a, e, h):
        # Oracle Alert!
        s = self.hm.state_map(None, None, h, g=self.g, q_func=self.oracle)
        s_nxt = self.hm.state_map(a, e, h, g=self.g, q_func=self.oracle)

        # curr_s = self.hm.state(history=h, extension=[a,e], level='current', *args)
        # next_s = self.hm.state(history=h, extension=[a,e], level='next', *args)
        # prev_s = self.hm.state(history=h, extension=[a,e], level='previous', *args)

        # update the reward matrix
        self.r[s][a][s_nxt] = (self.n[s][a][s_nxt]*self.r[s][a][s_nxt] + self.rm.r(a, e, h))/(self.n[s][a][s_nxt]+1)
        # update the transition matrix
        n_sum = self.n[s][a].sum()
        if n_sum: self.p[s][a] *= n_sum/(n_sum+1)
        self.p[s][a][s_nxt] += 1/(n_sum+1)
        # register the new input
        self.n[s][a][s_nxt] += 1

    def start(self, e = None):
        self.am.action = grl.epsilon_sample(self.am.actions)
        return self.am.action
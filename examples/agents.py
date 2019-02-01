import grl
import numpy as np
import math

class RandomAgent(grl.Agent):

    def act(self, h):
        return grl.epsilon_sample(self.am.actions)

    def learn(self, h, a, e):
        return

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
        self.Q.set_leaf_keys(self.am.action_space)

    def start(self, e=None, order=1):
        self.order = order
        self.am.action = grl.epsilon_sample(self.am.action_space)
        return self.am.action

    def act(self, h):
        s = self.hm.state(h)
        self.am.action = grl.epsilon_sample(self.am.action_space, max(self.Q[s])[1], 0.1)
        return self.am.action
    
    def learn(self, h, a, e):
        s = self.hm.state(h)
        s_nxt = self.hm.state(h, extension=[a,e], index=grl.Index.NEXT)
        r_nxt = self.rm.r(h, extension=[a,e], index=grl.Index.NEXT)
        
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
        self.xpl = self.kwargs.get('exploration_factor', 0.1)
        self.eps = self.kwargs.get('tolerance', 1e-6)
        self.steps = self.kwargs.get('steps', math.inf)
        self.v = grl.Storage(1, default=0)
        self.pi = grl.Storage(2, default=1)
        self.index = 0

    def interact(self, domain):
        super().interact(domain)
        self.pi.set_leaf_keys(self.am.action_space)
        self.pi.set_default(1/len(self.am.action_space))

    def act(self, h):
        self.pi, self.v = grl.PITabular(self.p, self.r, self.v, self.pi, g=self.g, steps=1, vi_steps=1)
        # Oracle Alert!
        s = self.hm.state(h, g=self.g, q_func=self.oracle)
        return grl.epsilon_sample(self.am.action_space, self.pi[s].argmax(), self.xpl)

    def learn(self, h, a, e):
        # Oracle Alert!
        s = self.hm.state(h, g=self.g, q_func=self.oracle)
        s_next = self.hm.state(h, extension=[a,e], index=grl.Index.NEXT, g=self.g, q_func=self.oracle)
        r_next = self.rm.r(h, extension=[a,e], index=grl.Index.NEXT)
        # update the reward matrix
        self.r[s][a][s_next] = (self.n[s][a][s_next]*self.r[s][a][s_next] + r_next)/(self.n[s][a][s_next]+1)
        # update the transition matrix
        n_sum = self.n[s][a].sum()
        if n_sum: self.p[s][a] *= n_sum/(n_sum+1)
        self.p[s][a][s_next] += 1/(n_sum+1)
        # register the new input
        self.n[s][a][s_next] += 1

    def start(self, e=None, order=1):
        self.order = order
        self.am.action = grl.epsilon_sample(self.am.action_space)
        return self.am.action

    def stats(self, *args, **kwargs):
        threshold = kwargs.get('significance_threshold', 0.1)
        n_total = self.n.sum()
        for s in self.n:
            print("{} = {:2.2f}%".format(s, 100*self.n[s].sum()/n_total))
            # if self.n[s].sum()/n_total > threshold:
            #     print("{} = {} {} {} {}".format(s, self.n[s].sum()/n_total, self.p[s], self.r[s], self.pi[s]))
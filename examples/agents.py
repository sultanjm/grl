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
        
        self.Q[s][a] = self.Q[s][a] + self.alpha[s][a] * (1-self.g) * (r_nxt + self.g * max(self.Q[s_nxt])[0] - self.Q[s][a])
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
        self.eps = self.kwargs.get('value_iteration_tolerance', 1e-6)
        self.steps = self.kwargs.get('value_iteration_max_steps', math.inf)
        self.v = grl.Storage(1, default=0)
        self.pi = grl.Storage(2, default=1)

    def interact(self, domain):
        super().interact(domain)
        self.pi.set_leaf_keys(self.am.actions)
        self.pi.set_default(1/len(self.am.actions))

    def act(self, h):
        self.v = ValueIteration(self.p, self.r, self.g, self.eps, self.v, 100)
        self.pi = PolicyImprovement(self.p, self.r, self.v, self.pi, self.g, 10)
        s = self.hm.state_map(None, None, h)
        return grl.epsilon_sample(self.am.actions, self.pi[s].argmax(), 1.0)

    def learn(self, a, e, h):
        s = self.hm.state_map(None, None, h)
        s_nxt = self.hm.state_map(a, e, h)
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

def phi_ExtremeVA(a, e, h):
    eps = 0.01
    v = optimal_value_of_the_domain(a, e, h)
    a_opt = optimal_action_of_the_domain(a, e, h)
    s = [np.floor(v/eps).astype('int'), a_opt]
    return s

def phi_ExtremeQ(a, e, h):
    eps = 0.01
    q = optimal_action_value_of_the_domain(a, e, h)
    s = np.floor(q/eps).astype('int')
    return s

def ValueIteration(p, r, g=0.999, eps=1e-6, v=None, steps=math.inf):
    if not isinstance(v, grl.Storage):
        v = grl.Storage(1, default=0, leaf_keys=p.keys())
    done = False
    while steps and not done:
        delta = 0
        for s in p:
            v_old = v[s]
            v[s] = (1 - g) * max([(r[s][a] + g * v).avg(p[s][a]) for a in p[s]])
            delta = max(delta, abs(v_old - v[s]))
        if delta < eps:
            done = True
        steps -= 1
    return v    

def PolicyIteration(p, r, g=0.999, eps=1e-6, v=None, pi=None, steps=math.inf):
    if not isinstance(pi, grl.Storage):
        pi = grl.Storage(2, default=1)
    v = ValueIteration(p, r, g, eps, v, steps)
    pi = PolicyImprovement(p, r, v, pi, steps)
    return pi

def PolicyImprovement(p, r, v, pi, g=0.999, steps=math.inf):
    stable = False
    while steps and not stable:
        stable = True
        for s in p:
            p_old = pi[s]
            v_s = {a:(r[s][a] + g * v).avg(p[s][a]) for a in p[s]}
            a_max = max(v_s, key=v_s.get)
            pi[s].clear()
            pi[s][a_max] = 1
            if pi[s] != p_old:
                stable = False
        steps -= 1
    return pi
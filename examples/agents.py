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
                             default_values=self.kwargs.get('value_function_init', (0,1)), 
                             persist=self.kwargs.get('value_function_persist', False))
        self.alpha = grl.Storage(dimensions=2, 
                                 default_values=self.kwargs.get('learning_rate_init', (0.999, 0.999)))

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
        self.n = grl.Storage(3, default_values=(1,1))
        self.r = grl.Storage(3, default_values=(0,0))

    def act(self, h):
        s = 1
        a = 2
        
        p_sa = self.n[s][a] / sum(self.n[s][a].values())
        r_sa = self.r[s][a] / sum(self.n[s][a].values())

    def learn(self, a, e, h):
        s = self.hm.state_map(h=h)
        s_nxt = self.hm.state_map(a, e, h)
        self.n[s][a][s_nxt] += 1
        self.r[s][a][s_nxt] += self.rm.r(a, e, h)

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
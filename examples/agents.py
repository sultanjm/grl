import grl
import numpy as np

class RandomAgent(grl.Agent):

    def setup(self):
        pass

    def act(self, percept):
        return grl.epsilon_sample(self.am.actions)

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

    def start(self, *args, **kwargs):
        self.am.action = grl.epsilon_sample(self.am.actions)
        return self.am.action
        

    # e = e_{t}
    # h = h_{t-1}
    # s = s_{t-1}
    # a = a_{t-1}
    # hae = h_{t}
    # s_nxt = s_{t}

    # h (s) --- a --> e (s_nxt)

    def act(self, e_nxt):
        a = self.am.action
        s = self.hm.mapped_state()
        s_nxt = self.hm.mapped_state(a, e_nxt)
        r_nxt = self.rm.r(a, e_nxt, self.hm.history)
        
        self.Q[s][a] = self.Q[s][a] + self.alpha[s][a] * (1-self.g) * (r_nxt + self.g * max(self.Q[s_nxt])[0] - self.Q[s][a])
        self.alpha[s][a] = self.alpha[s][a] * 0.999

        a_nxt = grl.epsilon_sample(self.am.actions, max(self.Q[s])[1], 0.1)

        self.am.action = a_nxt

        if self.keep_history: self.hm.history.append(a).append(e_nxt)

        return a_nxt

class StateAgent(grl.Agent):
    pass

class ExtremeQAgent(grl.Agent):
    pass

class ExtremeVAgent(grl.Agent):
    pass

def phi_EVA(h):
    v = optimal_value_of(h)
    a = optimal_action_of(h)

def phi_EQA(h):
    q = optimal_action_value_of(h)

class BinaryExtremeQAgent(grl.Agent):
    def setup(self):
        self.epsilon = self.kwargs.get('exploration_factor', 0.1)
        self.g = self.kwargs.get('discount_factor', 0.999)
        self.binary_actions = self.kwargs.get('binary_actions', [0,1])
        self.Q = grl.Storage(dimensions=2, 
                             default_values=self.kwargs.get('value_function_init', (0,1)), 
                             persist=self.kwargs.get('value_function_persist', False), 
                             leaf_keys=self.binary_actions)
        self.alpha = grl.Storage(dimensions=2, 
                                 default_values=self.kwargs.get('learning_rate_init', (0.999, 0.999)))
        self.r_bot = self.kwargs.get('r_bot', 0)
        self.am_b = grl.ActionManager(self.binary_actions, np.random.choice(self.binary_actions))
        self.ext_actions = self.binary_actions
        self.d = 0

    def interact(self, domain):
        super().interact(domain)
        actions = self.am.actions
        self.d = int(np.ceil(np.log2(len(actions))))
        diff = len(actions) - 2**self.d
        self.ext_actions = actions
        for _ in range(diff):
            self.ext_actions.append(actions[0])

    def action_map(self, a):
        pass
        
    def inv_action_map(self, b):
        pass
        
    def act(self, e):
        a = self.am.action
        # start internal loop over the binarized action space

        # This is not the first percept received from the domain.
        # This part looks ugly.
        if a:
            s = self.hm.mapped_state()
            s_nxt = self.hm.mapped_state(a, e)
            r_nxt = self.r(a, e, self.hm.history, idx)

            self.Q[s][a] = self.Q[s][a] + self.alpha[s][a] * (1-self.g) * (r_nxt + self.g * max(self.Q[s_nxt])[0] - self.Q[s][a])
            self.alpha[s][a] = self.alpha[s][a] * 0.999
        else:
            s = e

        self.am.action = grl.epsilon_sample(self.am.actions, max(self.Q[s])[1], 0.1)

        return self.am.action

    def r(self, a, e, h, idx):
        if idx < self.d:
            return self.r_bot
        return self.rm.r(a, e, h)
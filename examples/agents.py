import grl

class RandomAgent(grl.Agent):
    def act(self, percept):
        return grl.epsilon_sample(self.am.actions)

class GreedyQAgent(grl.Agent):
    def setup(self):
        self.epsilon = self.kwargs.get('exploration_factor', 0.1)
        self.g = self.kwargs.get('discount_factor', 0.999)
        self.Q = grl.Storage(dimensions=2, default_values=self.kwargs.get('Q_init', (0,1)), persist=self.kwargs.get('Q_persist', False))
        self.alpha = grl.Storage(dimensions=2, default_values=self.kwargs.get('learning_rate_init', (0.999, 0.999)))
    
    def interact(self, domain):
        super().interact(domain)
        self.Q.set_leaf_keys(self.am.actions)

    def act(self, e):

        a = self.am.action

        # This is not the first percept received from the domain.
        # This part looks ugly.
        if a:
            s = self.hm.mapped_state()
            s_nxt = self.hm.mapped_state(a, e)
            r_nxt = self.rm.r(a, e, self.hm.history)

            self.Q[s][a] = self.Q[s][a] + self.alpha[s][a] * (1-self.g) * (r_nxt + self.g * max(self.Q[s_nxt])[0] - self.Q[s][a])
            self.alpha[s][a] = self.alpha[s][a] * 0.999
        else:
            s = e

        self.am.action = grl.epsilon_sample(self.am.actions, max(self.Q[s])[1], 0.1)

        return self.am.action

class BinaryAgent(grl.Agent):
    def setup(self):
        self.epsilon = self.kwargs.get('exploration_factor', 0.1)
        self.g = self.kwargs.get('discount_factor', 0.999)
        self.Q = grl.Storage(dimensions=2, default_values=self.kwargs.get('Q_init', (0,1)), persist=self.kwargs.get('Q_persist', False))
        self.alpha = grl.Storage(dimensions=2, default_values=self.kwargs.get('learning_rate_init', (0.999, 0.999)))
        self.sm = grl.StateManager(self.transition_func, 0)

    def transition_func(self, s, a):
        return grl.epsilon_sample(self.T[a][s])

    def interact(self, domain):
        super().interact(domain)
        self.Q.set_leaf_keys(self.am.actions)
        self.T = grl.random_probability_matrix(len(self.am.actions), self.kwargs.get('num_of_states', 1))

    def act(self, e):
        a = self.am.action
        # This is not the first percept received from the domain.
        # This part looks ugly.
        if a:
            s = self.hm.mapped_state()
            s_nxt = self.hm.mapped_state(a, e)
            r_nxt = self.rm.r(a, e, self.hm.history)

            self.Q[s][a] = self.Q[s][a] + self.alpha[s][a] * (1-self.g) * (r_nxt + self.g * max(self.Q[s_nxt])[0] - self.Q[s][a])
            self.alpha[s][a] = self.alpha[s][a] * 0.999
        else:
            s = e

        self.am.action = grl.epsilon_sample(self.am.actions, max(self.Q[s])[1], 0.1)

        return self.am.action

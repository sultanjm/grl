import grl

class GreedyQAgent(grl.Agent):
    def setup(self):
        self.epsilon = self.kwargs.get('exploration_factor', 0.1)
        self.g = self.kwargs.get('discount_factor', 0.999)
        self.Q = grl.Storage(dimensions=2, default_values=self.kwargs.get('Q_init', (0,1)), persist=self.kwargs.get('Q_persist', False), compute_statistics=True)
        self.alpha = grl.Storage(dimensions=2, default_values=self.kwargs.get('learning_rate_init', (0.999, 0.999)))
    
    def interact(self, domain):
        super().interact(domain)
        self.Q.set_default_arguments(self.am.actions)

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

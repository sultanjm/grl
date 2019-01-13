import grl
import math

def VITabular(T, r, V=None, policy=None, **kwargs):
    steps = kwargs.get('steps', math.inf)
    g = kwargs.get('g', 0.999)
    eps = kwargs.get('eps', 1e-6)
    
    if not isinstance(V, grl.Storage):
        V = grl.Storage(1, default=0, leaf_keys=T.keys())

    done = False
    while steps and not done:
        delta = 0
        for s in T:
            v_s_old = V[s]
            if not policy:
                V[s] = max([(r[s][a] + g * V).avg(T[s][a]) for a in T[s]])
            else:
                V[s] = (policy[s] * {a:(r[s][a] + g * V).avg(T[s][a]) for a in T[s]}).sum()
            delta = max(delta, abs(v_s_old - V[s]))
        if delta < eps:
            done = True
        steps -= 1
    return V 

def PITabular(T, r, V=None, policy=None, **kwargs):
    steps = kwargs.get('steps', math.inf)
    vi_steps = kwargs.get('vi_steps', math.inf)
    g = kwargs.get('g', 0.999)
    eps = kwargs.get('eps', 1e-6)

    if not isinstance(policy, grl.Storage):
        actions = set(a for s in T.keys() for a in T[s].keys())
        policy = grl.Storage(2, default=1/len(actions), leaf_keys=actions)

    if not isinstance(V, grl.Storage):
        V = grl.Storage(1, default=0, leaf_keys=T.keys())

    stable = False
    while steps and not stable:
        V = VITabular(T, r, V, policy, steps=vi_steps, g=g, eps=eps)
        stable = True
        for s in T:
            policy_s_old = policy[s]
            v_s = {a:(r[s][a] + g * V).avg(T[s][a]) for a in T[s]}
            a_max = max(v_s, key=v_s.get)
            policy[s].clear()
            policy[s][a_max] = 1
            if policy[s] != policy_s_old:
                stable = False
        steps -= 1
    return policy, V

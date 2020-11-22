import copy
class DPAgent():
    def __init__(self, mdp, discount_rate=1):
        self.mdp = mdp
        self.value_fn = [0] * self.mdp.num_states
        num_a = self.mdp.num_actions
        num_s = self.mdp.num_states
        # random policy
        self.policy = [[1/num_a for a in range(num_a)] for s in range(num_s)]
        self.discount_rate = discount_rate

    def evaluate_policy(self):
        values = [0]*self.mdp.num_states
        for s in range(self.mdp.num_states):
            values[s] = self.evaluate_policy_for_state(s)
            values[s] = round(values[s], 3)
        self.value_fn = copy.deepcopy(values)
    
    def evaluate_policy_for_state(self, s):
        assert s < self.mdp.num_states
        value_s = 0
        for a in range(self.mdp.num_actions):
            for transition in self.mdp.P[s][a]:
                # transition informs the next state, probability of it, reward and if state is terminal
                pr_s_ = transition[0] # prob of next state s_ given s and a
                s_ = transition[1] # value of next state s_ given s and a
                r_s_ = transition[2] # reward of next state s_ given s and a
                value_s += self.policy[s][a] * pr_s_ * (r_s_ + self.discount_rate * self.value_fn[s_])
        return value_s

    def policy_improvement():
        pass

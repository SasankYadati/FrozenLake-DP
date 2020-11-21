class DPAgent():
    def __init__(self, mdp, discount_rate=1):
        self.mdp = mdp
        self.value_fn = [0] * self.mdp.num_states
        num_a = self.mdp.num_actions
        num_s = self.mdp.num_states
        self.policy = [[1/num_a for a in range(num_a)] for s in range(num_s)]

    def evaluate_policy(self):
        for s in range(self.mdp.num_states):
            self.evaluate_policy_for_state(s)
            self.value_fn[s] = round(self.value_fn[s], 3)
    
    def evaluate_policy_for_state(self, s):
        assert s < self.mdp.num_states
        for a in range(self.mdp.num_actions):
            for transition in self.mdp.P[s][a]:
                # transition informs the next state, probability of it, reward and if state is terminal
                pr_s_ = transition[0] # prob of next state s_ given s and a
                s_ = transition[1] # value of next state s_ given s and a
                r_s_ = transition[2] # reward of next state s_ given s and a
                self.value_fn[s] += self.policy[s][a] * pr_s_ * (r_s_ + self.discount_rate * self.value_fn[s_])

    def policy_improvement():
        pass

if __name__ == '__main__':
    from MarkovDecisionProcess import MarkovDecisionProcess as MDP
    import gym
    env = gym.make('FrozenLake-v0')
    env.reset()
    mdp = MDP(env.observation_space.n, env.action_space.n, env.unwrapped.P)
    agent = DPAgent(mdp)
    num_iters = 10
    for _ in range(num_iters):
        agent.evaluate_policy()
    print(agent.value_fn)

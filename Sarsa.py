import gym
import numpy as np

#number of epochs
MAX_NUM_EPISODES = 12000
#number of movements per epoch
STEPS_PER_EPISODE = 200
EPSILON_MIN = 0.005
max_num_steps = MAX_NUM_EPISODES * STEPS_PER_EPISODE
#step for decay of random chance
EPSILON_DECAY = 500 * EPSILON_MIN / max_num_steps
ALPHA = 0.001  # Learning rate
GAMMA = 0.95 # Discount factor
#since the environment is continuous, need to bin
NUM_DISCRETE_BINS = 30  # Number of bins to Discretize each observation dim

#classes in () are what the class inherits from
class Q_Learner():
    def __init__(self, env):
        self.obs_shape = env.observation_space.shape
        self.obs_high = env.observation_space.high
        self.obs_low = env.observation_space.low
        self.obs_bins = NUM_DISCRETE_BINS
        self.bin_width = (self.obs_high - self.obs_low) / self.obs_bins
        self.action_shape = env.action_space.n
        # Create a multi-dimensional array (aka. Table) to represent the
        # Q-values
        self.Q = np.zeros((self.obs_bins + 1, self.obs_bins + 1,
                           self.action_shape))  # (31 x 31 x 3)
        self.alpha = ALPHA  # Learning rate
        self.gamma = GAMMA  # Discount factor
        #start the probability for random state at 1
        self.epsilon = 1.0

    def discretize(self, obs):
        #returns the correct bin the value is
        return tuple(((obs - self.obs_low) / self.bin_width).astype(int))

    def get_action(self, obs):
        discretized_obs = self.discretize(obs)
        # Epsilon-Greedy action selection
        #decay epsilon
        if self.epsilon > EPSILON_MIN:
            self.epsilon -= EPSILON_DECAY
        #balances amount of exploration the agent does, based on how large epsilon is
        if np.random.random() > self.epsilon:
            #returns the index of the highest reward, which cooresponds to the action
            return np.argmax(self.Q[discretized_obs])
        else:  # Choose a random action
            # randomly select from action space
            return np.random.choice([a for a in range(self.action_shape)])

    def learn(self, obs, action, reward, next_obs):
        discretized_obs = self.discretize(obs)
        discretized_next_obs = self.discretize(next_obs)
        # in q function, the reward function
        #Q-learning algorithm
        #td_target = reward + self.gamma * np.max(self.Q[discretized_next_obs])

        #SARSA
        #get Q(S`, A`) for the Q update
        #can use either the epsilon-greedy action or the policy action - chose the policy action
        next_action = np.argmax(self.Q[discretized_next_obs])
        #next_action = self.get_action(next_obs)

        td_target = reward + self.gamma * self.Q[discretized_next_obs][next_action]

        td_error = td_target - self.Q[discretized_obs][action]
        #update the Q table
        self.Q[discretized_obs][action] += self.alpha * td_error

def train(agent, env):
    best_reward = -float('inf')
    for episode in range(MAX_NUM_EPISODES):
        done = False
        obs = env.reset()
        total_reward = 0.0
        while not done:
            action = agent.get_action(obs)
            #runs agent through environment
            next_obs, reward, done, info = env.step(action)
            #learn based of off action through env
            agent.learn(obs, action, reward, next_obs)
            obs = next_obs
            total_reward += reward
        if total_reward > best_reward:
            best_reward = total_reward
        print("Episode#:{} reward:{} best_reward:{} eps:{}".format(episode,
                                     total_reward, best_reward, agent.epsilon))

    # Return the trained policy, Q function
    return np.argmax(agent.Q, axis=2)

#test a policy (learned Q function) in env
def test(agent, env, policy):
    done = False
    obs = env.reset()
    total_reward = 0.0
    while not done:
        action = policy[agent.discretize(obs)]
        next_obs, reward, done, info = env.step(action)
        obs = next_obs
        total_reward += reward
    return total_reward


if __name__ == "__main__":
    env = gym.make('MountainCar-v0')
    agent = Q_Learner(env)
    learned_policy = train(agent, env)
    print(learned_policy.shape)
    print(learned_policy)
    # Use the Gym Monitor wrapper to evalaute the agent and record video
    gym_monitor_path = "./gym_monitor_output"
    env = gym.wrappers.Monitor(env, gym_monitor_path, force=True)
    for _ in range(1000):
        test(agent, env, learned_policy)
    env.close()
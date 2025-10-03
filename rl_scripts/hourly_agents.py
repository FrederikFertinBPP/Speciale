import numpy as np
from collections import deque
import random

import torch
from rl_scripts.agent import Agent, train
from RFP_environment import make_rfp_env
from rl_scripts.torch_networks import TorchActorNetwork, TorchCriticNetwork

class Scaler:
    """ Scales given data to either a normal distribution with mean 0 and sd 1,
    or a mapping from 0 and 1 with 0 being either a given lower bound or the min value of the data,
    and similarly either be given an upper bound or the max value of the data to map between.
    the scaler does not automatically clip the data between lower and upper bound, but instead provides values outside 0 and 1. """
    mu = None
    std = None

    def __init__(self, data):
        self.original_data = data
        pass

    def __call__(self):
        return self.scaled_data


class BasicBuffer:
    """
    The buffer class is used to keep track of past experience and sample it for learning.
    """
    def __init__(self, max_size=2000):
        """
        Creates a new (empty) buffer.

        :param max_size: Maximum number of elements in the buffer. This should be a large number like 100'000.
        """
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        """
        Add information from a single step, :math:`(s_t, a_t, r_{t+1}, s_{t+1}, \\text{done})` to the buffer.

        .. runblock:: pycon

            >>> import gymnasium as gym
            >>> from irlc.ex13.buffer import BasicBuffer
            >>> env = gym.make("CartPole-v1")
            >>> b = BasicBuffer()
            >>> s, info = env.reset()
            >>> a = env.action_space.sample()
            >>> sp, r, done, _, info = env.step(a)
            >>> b.push(s, a, r, sp, done)
            >>> len(b) # Get number of elements in buffer

        :param state: A state :math:`s_t`
        :param action: Action taken :math:`a_t`
        :param reward: Reward obtained :math:`r_{t+1}`
        :param next_state: Next state transitioned to :math:`s_{t+1}`
        :param done: ``True`` if the environment terminated else ``False``
        :return: ``None``
        """
        if type(reward) is not torch.Tensor: reward = np.array([reward])
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        """
        Sample ``batch_size`` elements from the buffer for use in training a deep Q-learning method.
        The elements returned all be numpy ``ndarray`` where the first dimension is the batch dimension, i.e. of size
        ``batch_size``.

        .. runblock:: pycon

            >>> import gymnasium as gym
            >>> from irlc.ex13.buffer import BasicBuffer
            >>> env = gym.make("CartPole-v1")
            >>> b = BasicBuffer()
            >>> s, info = env.reset()
            >>> a = env.action_space.sample()
            >>> sp, r, done, _, _ = env.step(a)
            >>> b.push(s, a, r, sp, done)
            >>> S, A, R, SP, DONE = b.sample(batch_size=32)
            >>> S.shape # Dimension batch_size x n
            >>> R.shape # Dimension batch_size x 1

        :param batch_size: Number of elements to sample
        :return:
            - S - Matrix of size ``batch_size x n`` of sampled states
            - A - Matrix of size ``batch_size x n`` of sampled actions
            - R - Matrix of size ``batch_size x n`` of sampled rewards
            - SP - Matrix of size ``batch_size x n`` of sampled states transitioned to
            - DONE - Matrix of size ``batch_size x 1`` of bools indicating if the environment terminated

        """
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []
        assert len(self.buffer) > 0, "The replay buffer must be non-empty in order to sample a batch: Use push()"
        batch = random.choices(self.buffer, k=batch_size)
        for state, action, reward, next_state, done in batch:
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return map(lambda x: np.asarray(x), (state_batch, action_batch, reward_batch, next_state_batch, done_batch))

    def sample_tensor(self, batch_size):
        """
        Sample ``batch_size`` elements from the buffer for use in training a deep Q-learning method.
        The elements returned all be numpy ``ndarray`` where the first dimension is the batch dimension, i.e. of size
        ``batch_size``.

        .. runblock:: pycon

            >>> import gymnasium as gym
            >>> from irlc.ex13.buffer import BasicBuffer
            >>> env = gym.make("CartPole-v1")
            >>> b = BasicBuffer()
            >>> s, info = env.reset()
            >>> a = env.action_space.sample()
            >>> sp, r, done, _, _ = env.step(a)
            >>> b.push(s, a, r, sp, done)
            >>> S, A, R, SP, DONE = b.sample(batch_size=32)
            >>> S.shape # Dimension batch_size x n
            >>> R.shape # Dimension batch_size x 1

        :param batch_size: Number of elements to sample
        :return:
            - S - Matrix of size ``batch_size x n`` of sampled states
            - A - Matrix of size ``batch_size x n`` of sampled actions
            - R - Matrix of size ``batch_size x n`` of sampled rewards
            - SP - Matrix of size ``batch_size x n`` of sampled states transitioned to
            - DONE - Matrix of size ``batch_size x 1`` of bools indicating if the environment terminated

        """
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []
        assert len(self.buffer) > 0, "The replay buffer must be non-empty in order to sample a batch: Use push()"
        batch = random.choices(self.buffer, k=batch_size,)
        for state, action, reward, next_state, done in batch:
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return map(lambda x: torch.stack(x) if type(x[0]) is torch.Tensor else np.asarray(x), (state_batch, action_batch, reward_batch, next_state_batch, done_batch))

    def __len__(self):
        return len(self.buffer)


class RandomAgent(Agent):
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def pi(self, s, k, info=None):
        return self.action_space.sample()  # Random action for simplicity


class DDPGAgent(Agent):
    def __init__(self, env, buffer=None, gamma=0.99, epsilon=None, alpha=0.001, batch_size=16,
                    hidden=30, replay_buffer_size=5000, replay_buffer_minreplay=500):
        super().__init__(env)
        # Ensure 'epsilon' is a function to allow gradually decreasing exploration rate
        self.epsilon        = epsilon if callable(epsilon) else lambda steps, episodes: epsilon
        self.gamma          = gamma
        # Initialize the replay buffer
        self.batch_size     = batch_size
        self.replay_buffer_minreplay = replay_buffer_minreplay
        self.memory         = BasicBuffer(replay_buffer_size) if buffer is None else buffer 
        # Initialize the actor and critic networks
        self.critic         = TorchCriticNetwork(env, trainable=True, learning_rate=alpha, hidden=hidden)
        self.critic_target  = TorchCriticNetwork(env, trainable=False, learning_rate=alpha, hidden=hidden)  # Target network
        self.critic_target.update_Phi(self.critic, tau=1.0)  # Initialize target network to match critic
        self.actor          = TorchActorNetwork(env, trainable=True, learning_rate=alpha, hidden=hidden)
        self.actor_target   = TorchActorNetwork(env, trainable=False, learning_rate=alpha, hidden=hidden)  # Target network
        self.actor_target.update_Phi(self.actor, tau=1.0)  # Initialize target network to match actor
        # Initialize step and episode counters
        self.steps, self.episodes = 0, 0

    def pi(self, s, k, info_s=None):
        """
        Compute the action to take in state :math:`s` using the actor network.
        :param s: Current state
        :param k: Not used in this implementation
        :param info_s: Additional information about the state (not used here)
        :return: Action to take, possibly with added noise for exploration
        """
        s = torch.FloatTensor(s)
        eps_ = self.epsilon(self.steps, self.episodes)  # Get epsilon value for exploration
        return self.actor(s) + eps_ * np.random.randn(self.actor.num_actions)  # Add noise for exploration

    def train(self, s, a, r, sp, done=False, info_s=None, info_sp=None):
        self.memory.push(torch.FloatTensor(s), 
                         torch.FloatTensor(a), 
                         torch.FloatTensor([r]),
                         torch.FloatTensor(sp),
                         torch.FloatTensor([int(done)]),
                         ) # save current observation 
        if len(self.memory) > self.replay_buffer_minreplay:
            self.experience_replay() # do the actual training step
        self.steps, self.episodes = self.steps + 1, self.episodes + done

    def experience_replay(self):
        s,a,r,sp,done = self.memory.sample_tensor(self.batch_size)
        # Update critic network
        y = r + self.gamma * self.critic_target.forward(sp, self.actor_target.forward(sp)) * (1 - done)  # Compute target Q-values
        self.critic.fit(y, s, a)  # Train the critic network
        # Update actor network
        self.actor.fit(self.critic, s)  # Train the actor network using the critic
        # Update target networks using Polyak averaging
        self.critic_target.update_Phi(self.critic, tau=0.01)
        self.actor_target.update_Phi(self.actor, tau=0.01)


if __name__ == "__main__":
    env = make_rfp_env(normalize=False)
    agent = RandomAgent(env)
    epsilon = lambda steps, episodes: max(0.1, 0.5 - steps / 10000)  # Epsilon decay function
    env_normed = make_rfp_env(normalize=True)
    agent_ddpg = DDPGAgent(env_normed, alpha=0.001, epsilon=epsilon, batch_size=16, replay_buffer_size=30000, hidden=10, depth=1)
    num_episodes = 8

    stats, trajectories = train(env, agent, num_episodes=num_episodes, experiment_name="random_test") # Quick
    stats_dqn, trajectories_dqn = train(env_normed, agent_ddpg, num_episodes=num_episodes, experiment_name="normalized_test") # 90 seconds/episode for now (also 37?)
    print("Training completed.")


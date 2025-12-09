from collections import deque
import random
import numpy as np
import pandas as pd
import torch

from common_scripts.agent import Agent
from model_scripts_old.torch_networks import TorchActorNetwork, TorchCriticNetwork, TorchActorWarmstarter, TorchActorNetworkSteering


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

    def sample(self, batch_size, as_tensor=False):
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
        if as_tensor:
            return map(lambda x: torch.stack(x) if type(x[0]) is torch.Tensor else np.asarray(x),
                       (state_batch, action_batch, reward_batch, next_state_batch, done_batch))
        else:
            return map(lambda x: np.asarray(x),
                       (state_batch, action_batch, reward_batch, next_state_batch, done_batch))

    def sample_tensor(self, batch_size):
        return self.sample(batch_size=batch_size, as_tensor=True)

    def __len__(self):
        return len(self.buffer)


class SARSABuffer(BasicBuffer):
    def push(self, *args):
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
        self.buffer.append(args)

    def sample(self, batch_size, as_tensor=False):
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
        prev_state_batch = []
        prev_action_batch = []
        prev_reward_batch = []
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []
        assert len(self.buffer) > 0, "The replay buffer must be non-empty in order to sample a batch: Use push()"
        batch = random.choices(self.buffer, k=batch_size)
        for prev_state, prev_action, prev_reward, state, action, reward, next_state, done in batch:
            prev_state_batch.append(prev_state)
            prev_action_batch.append(prev_action)
            prev_reward_batch.append(prev_reward)
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)
        if as_tensor:
            return map(lambda x: torch.stack(x) if type(x[0]) is torch.Tensor else np.asarray(x), 
                       (prev_state_batch, prev_action_batch, prev_reward_batch, state_batch, action_batch, reward_batch, next_state_batch, done_batch))

        else:
            return map(lambda x: np.asarray(x),
                   (prev_state_batch, prev_action_batch, prev_reward_batch, state_batch, action_batch, reward_batch, next_state_batch, done_batch))


class RandomAgent(Agent):
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def pi(self, s, k, info=None):
        return self.action_space.sample()  # Random action for simplicity


class DDPGAgent(Agent):
    def __init__(self,
                 env,                   # Gym environment
                 writer=None,           # Tensorboard agent writer
                 buffer=None,           # Object, which acts as the memory for the training procedure 
                 gamma=0.99,            # Discount factor of the future
                 epsilon=None,          # Exploration rate (scalar or a function dependent on steps and episodes)
                 alpha=0.001,           # Learning rate of the neural networks
                 batch_size=16,         # Batch size (number of environment interactions) used for each training iteration
                 hidden_size=30,        # Size of layers in neural networks.
                 replay_buffer_size=20000, # Number of environment interactions stored in the memory (buffer)
                 replay_buffer_minreplay=200, # Minimum number of environment interactions stored in the memory (buffer) before training is done.
                 tau=0.01,                # Update size of the target networks, which help define the gradients for the neural networks
                 sigma=0.5,               # Scalar of white noise sampled from a standard normal distribution
                 warmstarting_agent=None, # Warmstarter for the DDPG
                 **kwargs,
                 ):
        super().__init__(env, writer)
        self.num_actions = np.prod(env.action_space.shape)

        # Ensure 'epsilon' is a function to allow gradually decreasing exploration rate
        self.epsilon        = epsilon if callable(epsilon) else lambda steps, episodes: epsilon
        self.gamma          = gamma
        self.tau            = tau
        self.alpha          = alpha
        self.sigma          = sigma

        # Initialize the replay buffer
        self.batch_size     = batch_size
        self.replay_buffer_minreplay = replay_buffer_minreplay
        self.memory         = BasicBuffer(replay_buffer_size) if buffer is None else buffer
        
        # Initialize step and episode counters
        self.steps, self.episodes = 0, 0
        critic_warmstarter, actor_warmstarter = None, None
        self._define_context()
        if warmstarting_agent is not None:
            critic_warmstarter = warmstarting_agent.critic
            actor_warmstarter = warmstarting_agent.actor
            self.steps = warmstarting_agent.steps
        self._create_actor_critic(hidden_size, critic_warmstarter, actor_warmstarter, **kwargs)
    
    def _get_obs(self, s, k, info_s):
        """ Should return the observation, which the actor uses to inform its action. """
        t = info_s["time"]
        day_of_year = t.day_of_year
        state = [day_of_year] + list(s)
        return np.asarray(state)

    def _define_context(self):
        """ Should define the context space which is part of the observation in addition to the state. 
        Default is an empty context space. """
        self.context_space = np.asarray(['time'])
        self.context_size = np.prod(self.context_space.shape)

    def _get_action_noise(self):
        return np.random.randn(self.num_actions)

    def _create_actor_critic(self, hidden_size, *args, **kwargs):
        """ Should create the actor (Pi-function) and critic (Q-function) of the DDPG Agent. """
        self.actor  = lambda obs : super().pi()
        self.critic = lambda obs, action : 0

    def pi(self, s, k, info_s=None):
        """
        Compute the action to take in state :math:`s` using the actor network.
        :param s: Current state
        :param k: Not used in this implementation
        :param info_s: Additional information about the state (not used here)
        :return: Action to take, possibly with added noise for exploration
        """
        obs = self._get_obs(s, k, info_s)
        obs = torch.FloatTensor(obs)
        noise = self._get_action_noise() # Update noise terms
        eps_ = self.epsilon(self.steps, self.episodes)  # Get epsilon value for exploration
        actions = self.actor(obs) + eps_ * noise  # Add noise for exploration
        return actions.reshape(self.env.action_space.shape)


class OUDDPGAgent(DDPGAgent):
    def __init__(self,
                 env,                   # Gym environment
                 writer=None,           # Tensorboard agent writer
                 buffer=None,           # Object, which acts as the memory for the training procedure 
                 gamma=0.99,            # Discount factor of the future
                 epsilon=None,          # Exploration rate (scalar or a function dependent on steps and episodes)
                 alpha=0.001,           # Learning rate of the neural networks
                 batch_size=16,         # Batch size (number of environment interactions) used for each training iteration
                 hidden_size=30,        # Size of layers in neural networks.
                 replay_buffer_size=20000, # Number of environment interactions stored in the memory (buffer)
                 replay_buffer_minreplay=200, # Minimum number of environment interactions stored in the memory (buffer) before training is done.
                 tau=0.01,                # Update size of the target networks, which help define the gradients for the neural networks
                 sigma=0.5,               # Scalar of white noise sampled from a standard normal distribution for Ornstein-Uhlenbeck process
                 theta=0.9,               # Autoregressive factor of Ornstein-Uhlenbeck process.
                 warmstarting_agent=None, # Warmstarter for the DDPG
                 **kwargs,
                 ):
        super().__init__(env, writer, buffer, gamma, epsilon, alpha, batch_size, hidden_size, replay_buffer_size, replay_buffer_minreplay, tau, sigma, warmstarting_agent, **kwargs)
        
        # Parameters for Ornstein-Uhlenbeck process:
        self.mu = 0 # Mean of noise
        self.theta = theta
        self.x = np.zeros(self.num_actions) # Current noise term
    
    def _get_action_noise(self):
        # Autoregressive noise process - If we multiply by a time delta, then it is an OU process, which is a random walk in continuous time
        self.x += self.theta * (self.mu - self.x) + self.sigma * np.random.randn(self.num_actions)
        return self.x


class StateAwareDDPGAgent(OUDDPGAgent):
    def _create_actor_critic(self, hidden_size, critic_warmstarter=None, actor_warmstarter=None, **kwargs):
        # Initialize the actor and critic networks
        self.critic         = TorchCriticNetwork(self.env, trainable=True, learning_rate=self.alpha, hidden=hidden_size, context_size=self.context_size, writer=self.writer)
        if critic_warmstarter is not None:
            self.critic.update_Phi(critic_warmstarter, tau=1)
        self.critic_target  = TorchCriticNetwork(self.env, trainable=False, learning_rate=self.alpha, hidden=hidden_size, context_size=self.context_size)  # Target network
        self.critic_target.update_Phi(self.critic, tau=1.0)  # Initialize target network to match critic
        self.actor          = TorchActorNetwork(self.env, trainable=True, learning_rate=self.alpha, hidden=hidden_size, context_size=self.context_size, writer=self.writer)
        if actor_warmstarter is not None:
            self.actor.update_Phi(actor_warmstarter, tau=1)
        self.actor_target   = TorchActorNetwork(self.env, trainable=False, learning_rate=self.alpha, hidden=hidden_size, context_size=self.context_size)  # Target network
        self.actor_target.update_Phi(self.actor, tau=1.0)  # Initialize target network to match actor

    def save(self, path):
        self.critic.save(path + "/critic_network")
        self.actor.save(path + "/actor_network")
        self.critic_target.save(path + "/critictarget_network")
        self.actor_target.save(path + "/actortarget_network")
    
    def load(self, path):
        self.critic.load(path + "/critic_network")
        self.actor.load(path + "/actor_network")
        self.critic_target.load(path + "/critictarget_network")
        self.actor_target.load(path + "/actortarget_network")

    def train(self, s, a, r, sp, done=False, info_s=None, info_sp=None):
        obs = self._get_obs(s=s, k=0, info_s=info_s)
        obs_p = self._get_obs(s=sp, k=0, info_s=info_sp)
        self.memory.push(torch.FloatTensor(obs), 
                         torch.FloatTensor(a.reshape(-1)), 
                         torch.FloatTensor([r]),
                         torch.FloatTensor(obs_p),
                         torch.FloatTensor([int(done)]),
                         ) # save current observation
        if len(self.memory) > self.replay_buffer_minreplay:
            self.experience_replay() # do the actual training step
        self.steps = self.steps + 1
        if done:
            self.episodes += 1
            self.actor.epoch = self.episodes
            self.critic.epoch = self.episodes

    def experience_replay(self):
        """ Performs Q-learning of the neural networks. """
        obs,a,r,obs_p,done = self.memory.sample_tensor(self.batch_size)
        # Update critic network
        y = r + self.gamma * self.critic_target.forward(obs_p, self.actor_target.forward(obs_p)) * (1 - done)  # Compute target Q-values
        self.critic.fit(y, obs, a)  # Train the critic network
        # Update actor network
        self.actor.fit(self.critic, obs)  # Train the actor network using the critic
        # Update target networks using Polyak averaging
        self.critic_target.update_Phi(self.critic, tau=self.tau)
        self.actor_target.update_Phi(self.actor, tau=self.tau)


class ContextAwareDDPGAgent(StateAwareDDPGAgent):
    def _define_context(self):
        self.context_space = np.asarray(['time'])
        self.context_space = np.asarray([
            ['wind'] * self.env.decision_horizon, 
            ['solar'] * self.env.decision_horizon,
            ['price'] * self.env.decision_horizon,
        ])
        self.context_size = np.prod(self.context_space.shape) + 1
    
    def _get_obs(self, s, k, info_s):
        t = info_s["time"]
        day_of_year = t.day_of_year
        state = [day_of_year] + list(s)

        wind_production = info_s['asset_wind_realization']['wind']
        solar_production = info_s['asset_solar_realization']['solar']
        forecasts = self.env.forecaster.forecast(start=t, end=t+pd.Timedelta(self.env.decision_horizon-1, 'h'), n_forecasts=1)
        price_profile = forecasts[0]['price']
        context = list(wind_production) + list(solar_production) + list(price_profile)
        
        return np.asarray(state + context)


class WarmStartDDPGAgent(ContextAwareDDPGAgent):
    def __init__(self, env, buffer=None, gamma=0.99, epsilon=None, alpha=0.001, batch_size=16, hidden_size=30, replay_buffer_size=20000, replay_buffer_minreplay=200, tau=0.01):
        buffer = SARSABuffer(replay_buffer_size)
        super().__init__(env, buffer, gamma, epsilon, alpha, batch_size, hidden_size, replay_buffer_size, replay_buffer_minreplay, tau)
        self.previous_action = None # env.action_space.sample()
        self.previous_obs = None # self._get_obs(s=env.state_space.low,k=0,info_s=env.context)
        self.previous_r = None # 0

    def _get_obs(self, s, k, info_s=None, info_sp=None):
        t = info_s["time"]
        wind_production = info_s['asset_wind_realization']['wind']
        solar_production = info_s['asset_solar_realization']['solar']
        if info_sp is not None:
            price_profile = info_sp['electricity_price']
        else:
            price_profile = info_s['electricity_price']
        obs = np.asarray(list(s) + list(wind_production) + list(solar_production) + list(price_profile))
        return obs

    def _create_actor_critic(self, hidden_size, *args, **kwds):
        # Initialize the actor and critic networks
        self.critic         = TorchCriticNetwork(self.env, trainable=True, learning_rate=self.alpha, hidden=hidden_size, context_size=self.context_size)
        self.critic_target  = TorchCriticNetwork(self.env, trainable=False, learning_rate=self.alpha, hidden=hidden_size, context_size=self.context_size)  # Target network
        self.critic_target.update_Phi(self.critic, tau=1.0)  # Initialize target network to match critic
        self.actor          = TorchActorWarmstarter(self.env, trainable=True, learning_rate=self.alpha, hidden=hidden_size, context_size=self.context_size)
    
    def train(self, s, a, r, sp, done=False, info_s=None, info_sp=None):
        if self.previous_r is None:
            self.previous_action = a
            self.previous_obs = self._get_obs(s=s, k=0, info_s=info_s, info_sp=info_sp)
            self.previous_r = r
        else:
            obs = self._get_obs(s=s, k=0, info_s=info_s, info_sp=info_sp)
            next_obs = self._get_obs(s=sp, k=0, info_s=info_sp)
            self.memory.push(torch.FloatTensor(self.previous_obs),
                                torch.FloatTensor(self.previous_action.reshape(-1)),
                                torch.FloatTensor([self.previous_r]),
                                torch.FloatTensor(obs),
                                torch.FloatTensor(a.reshape(-1)),
                                torch.FloatTensor([r]),
                                torch.FloatTensor(next_obs),
                                torch.FloatTensor([int(done)]),
                                ) # save current observation SARSARS
            if len(self.memory) > self.replay_buffer_minreplay:
                self.experience_replay() # do the actual training step
            self.previous_action = a
            self.previous_obs = obs
            self.previous_r = r
        self.steps, self.episodes = self.steps + 1, self.episodes + done

    def experience_replay(self):
        """ Performs SARSA-learning of the neural networks. """
        prev_obs,prev_a,prev_r,obs,a,r,next_obs,done = self.memory.sample_tensor(self.batch_size)
        # Update critic network
        y = prev_r + self.gamma * (self.critic_target.forward(obs, a) * (1-done) + r * done)  # Compute target Q-values
        self.critic.fit(y, prev_obs, prev_a)  # Train the critic network
        # Update actor network
        self.actor.fit(prev_a, prev_obs)  # Train the actor network using the critic
        self.actor.fit(a, obs)  # Train the actor network using the critic
        # Update target network using Polyak averaging
        self.critic_target.update_Phi(self.critic, tau=self.tau)


class SteeringDDPGAgent(StateAwareDDPGAgent):
    def _create_actor_critic(self, hidden_size, critic_warmstarter=None, actor_warmstarter=None, initial_guess:np.ndarray|None = None, **kwargs):
        # Initialize the actor and critic networks
        self.critic         = TorchCriticNetwork(self.env, trainable=True, learning_rate=self.alpha, hidden=hidden_size, context_size=self.context_size, writer=self.writer)
        if critic_warmstarter is not None:
            self.critic.update_Phi(critic_warmstarter, tau=1)
        self.critic_target  = TorchCriticNetwork(self.env, trainable=False, learning_rate=self.alpha, hidden=hidden_size, context_size=self.context_size)  # Target network
        self.critic_target.update_Phi(self.critic, tau=1.0)  # Initialize target network to match critic
        self.actor          = TorchActorNetworkSteering(self.env, trainable=True, learning_rate=self.alpha, hidden=hidden_size, context_size=self.context_size, writer=self.writer)
        if initial_guess is not None:
            assert len(initial_guess) == self.num_actions, "The initial guess should be a vector with the size of the number of actions."
            self.actor.output_layer.bias.data = torch.tensor(initial_guess, dtype=torch.float32)
        if actor_warmstarter is not None:
            self.actor.update_Phi(actor_warmstarter, tau=1)
        self.actor_target   = TorchActorNetworkSteering(self.env, trainable=False, learning_rate=self.alpha, hidden=hidden_size, context_size=self.context_size)  # Target network
        self.actor_target.update_Phi(self.actor, tau=1.0)  # Initialize target network to match actor
    

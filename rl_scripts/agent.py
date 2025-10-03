# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""The Agent class.

References:
  [Her25] Tue Herlau. Sequential decision making. (Freely available online), 2025.
"""
import typing
import itertools
import os
import sys
from collections import OrderedDict, namedtuple
import numpy as np
from tqdm import tqdm

# import shutil
from gymnasium import Env
from dataclasses import dataclass
from utils import cache_write, Trajectory, fields

from gymnasium import Agent

class Agent: 
    r"""The main agent class. See (Her25, Subsection 4.4.3) for additional details.

    To use the agent class, you should first create an environment. In this case we will just create an instance of the
    ``InventoryEnvironment`` (see (Her25, Subsection 4.2.3))

    :Example:

        .. runblock:: pycon

            >>> from irlc import Agent                                              # You can import directly from top-level package
            >>> import numpy as np
            >>> np.random.seed(42)                                                  # Fix the seed for reproduciability
            >>> from irlc.ex01.inventory_environment import InventoryEnvironment
            >>> env = InventoryEnvironment()                                        # Create an instance of the environment
            >>> agent = Agent(env)                                                  # Create an instance of the agent.
            >>> s0, info0 = env.reset()                                             # Always call reset to start the environment
            >>> a0 = agent.pi(s0, k=0, info=info0)                                  # Tell the agent to compute action $a_{k=0}$
            >>> print(f"In state {s0=}, the agent took the action {a0=}")
    """
    
    def __init__(self, env: Env):
        """Instantiate the Agent class.

        The agent is given the openai gym environment it must interact with. This allows the agent to know what the
        action and observation space is.

        :param env: The openai gym ``Env`` instance the agent should interact with.
        """
        self.env = env   

    def pi(self, s, k : int, info : typing.Optional[dict] =None):
        r"""Evaluate the Agent's policy (i.e., compute the action the agent want to take) at time step ``k`` in state ``s``.
        
        This correspond to the environment being in a state evaluating :math:`x_k`, and the function should compute the next
        action the agent wish to take:
                
        .. math::
            u_k = \mu_k(x_k)
        
        This means that ``s`` = :math:`x_k` and ``k`` = :math:`k =\{0, 1, ...\}`. The function should return an action that lies in the action-space
        of the environment.
        
        The info dictionary:
            The ``info``-dictionary contains possible extra information returned from the environment, for instance when calling the ``s, info = env.reset()`` function.
            The main use in this course is in control, where the dictionary contains a value ``info['time_seconds']`` (which corresponds to the simulation time :math:`t` in seconds).
            
            We will also use the info dictionary to let the agent know certain actions are not available. This is done by setting the ``info['mask']``-key. 
            Note that this is only relevant for reinforcement learning, and you should see the documentation/exercises for reinforcement learning for additional details.
        
        The default behavior of the agent is to return a random action. An example:
        
        .. runblock:: pycon
        
            >>> from irlc.pacman.pacman_environment import PacmanEnvironment
            >>> from irlc import Agent
            >>> env = PacmanEnvironment()
            >>> s, info = env.reset()
            >>> agent = Agent(env)            
            >>> agent.pi(s, k=0, info=info) # get a random action
            >>> agent.pi(s, k=0)            # If info is not specified, all actions are assumed permissible.
                

        :param s: Current state the environment is in.
        :param timestep: Current time
        :return: The action the agent want to take in the given state at the given time. By default the agent returns a random action
        """ 
        if info is None or 'mask' not in info:
            return self.env.action_space.sample()
        else:
            """ In the case where the actions available in each state differ, openAI deals with that by specifying a 
            ``mask``-entry in the info-dictionary. The mask can then be passed on to the 
            env.action_space.sample-function to make sure we don't sample illegal actions. I consider this the most 
            difficult and annoying thing about openai gym."""
            if info['mask'].max() > 1:
                raise Exception("Bad mask!")
            return self.env.action_space.sample(mask=info['mask']) 


    def train(self, s, a, r, sp, done=False, info_s=None, info_sp=None): 
        r"""Implement this function if the agent has to learn (be trained).

        Note that you only have to implement this function from week 7 onwards -- before that, we are not interested in control methods that learn.
        
        The agent takes a number of input arguments. You should imagine that
         
        * ``s`` is the current state :math:`x_k``
        * ``a`` is the action the agent took in state ``s``, i.e. ``a`` :math:`= u_k = \mu_k(x_k)`
        * ``r`` is the reward the the agent got from that action
        * ``sp`` (s-plus) is the state the environment then transitioned to, i.e. ``sp`` :math:`= x_{k+1}`
        * '``done`` tells the agent if the environment has stopped
        * ``info_s`` is the information-dictionary returned by the environment as it transitioned to ``s``
        * ``info_sp`` is the information-dictionary returned by the environment as it transitioned to ``sp``.
          
        The following example will hopefully clarify it by showing how you would manually call the train-function once:
          
        :Example:      
           
            .. runblock:: pycon

                >>> from irlc.ex01.inventory_environment import InventoryEnvironment    # import environment
                >>> from irlc import Agent
                >>> env = InventoryEnvironment()                                        # Create an instance of the environment
                >>> agent = Agent(env)                                                  # Create an instance of the agent.
                >>> s, info_s = env.reset()                                             # s is the current state
                >>> a = agent.pi(s, k=0, info=info_s)                                   # The agent takes an action
                >>> sp, r, done, _, info_sp = env.step(a)                               # Environment updates
                >>> agent.train(s, a, r, sp, done, info_s, info_sp)                     # How the training function is called

        
        In control and dynamical programming, please recall that the reward is equal to minus the cost.
        
        :param s: Current state :math:`x_k`
        :param a: Action taken :math:`u_k`
        :param r: Reward obtained by taking action :math:`a_k` in state :math:`x_k`
        :param sp: The state that the environment transitioned to :math:`{\\bf x}_{k+1}`
        :param info_s: The information dictionary corresponding to ``s`` returned by ``env.reset`` (when :math:`k=0`) and otherwise ``env.step``.
        :param info_sp: The information-dictionary corresponding to ``sp`` returned by ``env.step``
        :param done: Whether environment terminated when transitioning to ``sp``
        :return: None
        """
        pass  

    def __str__(self):
        """**Optional:** A unique name for this agent. Used for labels when plotting, but can be kept like this."""
        return super().__str__()

    def extra_stats(self) -> dict:
        """**Optional:** Implement this function if you wish to record extra information from the ``Agent`` while training.

        You can safely ignore this method as it will only be used for control theory to create nicer plots """
        return {}

# Experiment using a dataclass.
@dataclass
class Stats:
    episode: int
    episode_length: int
    accumulated_reward: float

    total_steps: int
    trajectory : Trajectory = None
    agent_stats : dict = None

    @property
    def average_reward(self):
        return self.accumulated_reward / self.episode_length

# s = Stats(episode=0, episode_length=5, accumulated_reward=4, total_steps=2, trajectory=Trajectory())


def train(env,
          agent=None,
          experiment_name=None,
          num_episodes=1,
          verbose=True,
          max_steps=1e10,
          log_interval=1, # Only log every log_interval steps. Reduces size of log files.
          seed=None, # Attempt to set the seed of the random number generator to produce reproducible results.
          ):
    """This function implements the main training loop as described in (Her25, Subsection 4.4.4).

    The loop will simulate the interaction between agent `agent` and the environment `env`.
    The function has a lot of special functionality, so it is useful to consider the common cases. An example:

    >>> stats, _ = train(env, agent, num_episodes=2)

    Simulate interaction for two episodes (i.e. environment terminates two times and is reset).
    `stats` will be a list of length two containing information from each run

    >>> stats, trajectories = train(env, agent, num_episodes=2, return_Trajectory=True)

    `trajectories` will be a list of length two containing information from the two trajectories.

    >>> stats, _ = train(env, agent, experiment_name='experiments/my_run', num_episodes=2)

    Save `stats`, and trajectories, to a file which can easily be loaded/plotted (see course software for examples of this).
    The file will be time-stamped so using several calls you can repeat the same experiment (run) many times.

    >>> stats, _ = train(env, agent, experiment_name='experiments/my_run', num_episodes=2, max_runs=10)

    As above, but do not perform more than 10 runs. Useful for repeated experiments.

    :param env: An openai-Gym ``Env`` instance (the environment)
    :param agent: An ``Agent`` instance
    :param experiment_name: The outcome of this experiment will be saved in a folder with this name. This will allow you to run multiple (repeated) experiment and visualize the results in a single plot, which is very important in reinforcement learning.
    :param num_episodes: Number of episodes to simulate
    :param verbose: Display progress bar
    :param reset: Call ``env.reset()`` before simulation start. Default is ``True``. This is only useful in very rare cases.
    :param max_steps: Terminate if this many steps have elapsed (for non-terminating environments)
    :param max_runs: Maximum number of repeated experiments (requires ``experiment_name``)
    :param return_trajectory: Return trajectories list (Off by default since it might consume lots of memory)
    :param resume_stats: Resume stat collection from last run (this requires the ``experiment_name`` variable to be set)
    :param log_interval: Log stats less frequently than each episode. Useful if you want to run really long experiments.
    :param delete_old_experiments: If true, old saved experiments will be deleted. This is useful during debugging.
    :param seed: An integer. The random number generator of the environment will be reset to this seed allowing for reproducible results.
    :return: A list where each element corresponds to each (started) episode. The elements are dictionaries, and contain the statistics for that episode.
    """

    if agent is None:
        print("[train] No agent was specified. Using irlc.Agent(env) (this agent selects actions at random)")
        agent = Agent(env)

    stats = []
    steps = 0
    ep_start = 0
    trajectories = []
    break_outer = False

    with tqdm(total=num_episodes, disable=not verbose, file=sys.stdout, mininterval=int(num_episodes/100) if num_episodes>100 else None) as tq:
        for i_episode in range(num_episodes):
            if break_outer:
                break
            s, info_s = env.reset(seed=seed) # Reset the environment at the beginning of each episode. 

            reward = []
            trajectory = Trajectory(time=[], state=[], action=[], reward=[], env_info=[])
            time = 0 # initial time

            for _ in itertools.count():
                # Main train loop:
                a = agent.pi(s, time, info_s)
                time += 1
                sp, r, terminated, truncated, info_sp = env.step(a)
                done = terminated or truncated
                agent.train(s, a, r, sp, done, info_s, info_sp)

                # Logging the training progress:
                trajectory.time.append(np.asarray(info_s['time_seconds'] if 'time_seconds' in info_s else steps)) #np.asarray(time))
                trajectory.state.append(s)
                trajectory.action.append(a)
                trajectory.reward.append(np.asarray(r))
                trajectory.env_info.append(info_s)

                reward.append(r)
                steps += 1

                if done or steps >= max_steps:
                    trajectory.state.append(sp)
                    trajectory.env_info.append(info_sp)
                    trajectory.time.append(np.asarray(info_sp.get('time_seconds', None) if 'time_seconds' in info_s else steps))
                    break_outer = steps >= max_steps
                    break
                s = sp 
                info_s = info_sp
            
            # Saving the episode
            trajectories.append(trajectory)
            if (i_episode + 1) % log_interval == 0:
                stats.append({"Episode": i_episode + ep_start,
                              "Accumulated Reward": sum(reward),
                              "Length": len(reward),
                              "Steps": steps, # Useful for deep learning applications. This should be kept, or week 13 will have issues.
                              **agent.extra_stats()})

            rate = int(num_episodes / 100)
            if rate > 0 and i_episode % rate == 0:
                tq.set_postfix(ordered_dict=OrderedDict(list(OrderedDict(stats[-1]).items())[:5])) if len(stats) > 0 else None
            tq.update()

    if experiment_name:
        from time import time
        t = str(time()).split('.')[0]
        file_path = os.getcwd() + "/Speciale/experiments/" + experiment_name + "_tstamp_" + t + "/trajectories.pkl"
        cache_write(trajectories, file_path, verbose=verbose)
    # sys.stderr.flush()

    return stats, trajectories


# if __name__ == "__main__":
#     # Use the trajectories here.
#     from irlc.ex01.inventory_environment import InventoryEnvironment
#     env = InventoryEnvironment(N=10)
#     stats, traj = train(env, Agent(env))
#     print(stats)
#     s = Stats(episode=1, episode_length=2, accumulated_reward=4, total_steps=4, trajectory=None, agent_stats={})
#     print(s)

""" Script to plot trajectories from a saved file. """
from utils import cache_read
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from RFP_environment import make_hpp_env

def load_trajectories(experiment_name, ):
    # Get latest of the experiments if there are multiple of the same name:
    folder = os.getcwd() + "/Speciale/experiments/"
    experiments = os.listdir(folder)
    t = 0 # Very large time compared to unix time from time.time()
    for exp in experiments: # Load only the most recent experiment
        name, t_ = tuple(exp.split('_tstamp_'))
        t_ = int(t_)
        if name == experiment_name:
            if t_ > t:
                t = t_
    file_path = folder + experiment_name + "_tstamp_" + str(t) + "/trajectories.pkl"
    trajectories = cache_read(file_path)
    return trajectories

def plot_trajectory(trajectories, field:str, **kwargs):
    fields = ('time', 'state', 'action', 'reward', 'env_info')
    funcs = (lambda:print("Cannot plot time"),
             plot_state_trajectories,
             plot_action_trajectories,
             plot_reward_trajectory,
             plot_env_info, )
    kwargs['total_observations'] = sum(len(traj.reward) for traj in trajectories)
    kwargs['n_episodes'] = len(trajectories)

    for n, f_ in enumerate(fields):
        if field == f_:
            funcs[n](trajectories, **kwargs)

def plot_state_trajectories(trajectories, **kwargs):
    df = pd.DataFrame(columns=kwargs['state_names'], index=range(0, kwargs['total_observations'] + kwargs['n_episodes']))
    start_row = 0
    for traj in trajectories:
        last_row = start_row + len(traj.state)
        for ix, name in enumerate(kwargs['state_names']):
            if kwargs['plot_state'][ix]:
                data = np.transpose(np.array([x[ix] for x in traj.state]))
                df.loc[(df.index >= start_row) & (df.index < last_row), name] = data
        start_row = last_row
    sns.lineplot(df, alpha=1)
    plt.show()

def plot_action_trajectories(trajectories, **kwargs):
    df = pd.DataFrame(columns=kwargs['action_names'], index=range(1, kwargs['total_observations'] + 1))
    start_row = 1
    for traj in trajectories:
        info = traj.env_info[1:]
        last_row = start_row + len(info)
        for ix, name in enumerate(kwargs['action_names']):
            data = np.transpose(np.array([x['action'][ix] for x in info]))
            df.loc[(df.index >= start_row) & (df.index < last_row), name] = data
        start_row = last_row
    sns.lineplot(df, alpha=1)
    plt.show()

def plot_reward_trajectory(trajectories, **kwargs):
    df = pd.DataFrame(columns=['Reward'], index=range(1, kwargs['total_observations'] + 1))
    start_row = 1
    for traj in trajectories:
        last_row = start_row + len(traj.reward)
        df.loc[(df.index >= start_row) & (df.index < last_row), 'Reward'] = np.array(traj.reward)
        start_row = last_row
    sns.lineplot(df, alpha=1)
    plt.show()

def plot_env_info(trajectories, **kwargs):
    df = pd.DataFrame(columns=kwargs['env_info_keys'], index=range(1, kwargs['total_observations'] + 1))
    start_row = 1
    for traj in trajectories:
        info = traj.env_info[1:]
        last_row = start_row + len(info)
        for key in kwargs['env_info_keys']:
            try:
                data = np.transpose(np.array([x[key] for x in info]))
                df.loc[(df.index >= start_row) & (df.index < last_row), key] = data
            except(KeyError):
                raise("Please provide an env_info_key. Available keys are: ", info[0].keys())
            finally:
                continue
        start_row = last_row
    sns.lineplot(df)
    plt.show()

if __name__ == "__main__":
    env = make_hpp_env()
    trajectories = load_trajectories("normalized_test")
    # plot_trajectory(trajectories, 'env_info', **{'env_info_keys': ['wind_power', 'solar_power']})
    # plot_trajectory(trajectories, 'reward')
    # plot_trajectory(trajectories, 'state', **{'state_names': env.state_names, 'plot_state': np.ones(len(env.state_names))})
    plot_trajectory(trajectories, 'action', **{'action_names': env.action_names, 'plot_state': np.ones(len(env.action_names))})
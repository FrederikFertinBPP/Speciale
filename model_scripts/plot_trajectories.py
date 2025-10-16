""" Script to plot trajectories from a saved file. """
from common_scripts.utils import cache_read
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def load_trajectories(experiment_name, ):
    # Get latest of the experiments if there are multiple of the same name:
    folder = os.getcwd() + "/experiments/"
    experiments = os.listdir(folder)
    t = -10**6 # Very large time compared to unix time from time.time()
    exp_name = tuple(experiment_name.split('_'))
    for exp in experiments: # Load only the most recent experiment
        name = tuple(exp.split('_'))
        t_ = int(name[-1])
        name = name[:-2]
        if name == exp_name:
            if t_ > t:
                t = t_
    file_path = folder + experiment_name + "_runtime_" + str(t) + "/trajectories.pkl"
    trajectories = cache_read(file_path)
    return trajectories

def plot_trajectory(env, trajectory, field:str, **kwargs):
    fields = ('time', 'state', 'action', 'reward', 'env_info')
    funcs = (lambda:print("Cannot plot time"),
             plot_state_trajectory,
             plot_action_trajectory,
             plot_reward_trajectory,
             plot_env_info,)
    kwargs['total_observations'] = len(trajectory.reward)

    for n, f_ in enumerate(fields):
        if field == f_:
            funcs[n](env, trajectory, **kwargs)
    plot_name = kwargs.get('plot_name', None)
    if plot_name is not None:
        plt.savefig(f'trajectory_plots/{plot_name}.png')
        plt.close()
        return None
    else:
        return trajectory

def plot_state_trajectory(env, trajectory, **kwargs):
    plot_mask = kwargs.get("plot_mask" , np.ones(len(env.state_names)).astype(bool))
    columns = [name for name, plot in zip(env.state_names, plot_mask) if plot]
    df = pd.DataFrame(columns=columns, index=range(0, kwargs['total_observations'] + 1))
    for ix, name in enumerate(env.state_names):
        if plot_mask[ix]:
            data = np.transpose(np.array([x[ix] for x in trajectory.state]))
            df.loc[:, name] = data
    sns.lineplot(df, alpha=1)

def plot_action_trajectory(env, trajectory, **kwargs):
    if len(env.action_space.shape) == 2:
        T, N = env.action_space.shape # If matrix, then action has a time-index, which is the first dimension.
    else:
        N, = env.action_space.shape
        T = 1
    plot_mask = kwargs.get("plot_mask" , np.ones(N).astype(bool))
    start_row = 1
    columns = [name for name, plot in zip(env.action_names, plot_mask) if plot]
    df = pd.DataFrame(columns=columns, index=range(start_row, kwargs['total_observations']*T + start_row))
    for actions in trajectory.action:
        df.loc[(df.index >= start_row) & (df.index < start_row+T)] = actions[:, plot_mask]
        start_row += T
    sns.lineplot(df, alpha=1)

def plot_reward_trajectory(env, trajectory, **kwargs):
    df = pd.DataFrame(data={'Reward': np.array(trajectory.reward)}, index=range(1, kwargs['total_observations'] + 1))
    sns.lineplot(df, alpha=1)

def plot_env_info(env, trajectory, **kwargs):
    start_row = 1
    df = pd.DataFrame(columns=kwargs['env_info_keys'], index=range(start_row, kwargs['total_observations'] + start_row))
    info = trajectory.env_info[1:]
    for key in kwargs.get('env_info_keys', []):
        data = np.array([x[key] for x in info])[-kwargs['total_observations']:]
        if len(data.shape) == 3:
            print("Use action plotting function to plot actions.")
        if len(data.shape) == 2:
            data = np.mean(data, axis=1)
        df.loc[:, key] = data
    sns.lineplot(df)

# if __name__ == "__main__":
#     env = make_rfp_env()
#     trajectories = load_trajectories("normalized_test")
#     # plot_trajectory(trajectories, 'env_info', **{'env_info_keys': ['wind_power', 'solar_power']})
#     # plot_trajectory(trajectories, 'reward')
#     # plot_trajectory(trajectories, 'state', **{'state_names': env.state_names, 'plot_state': np.ones(len(env.state_names))})
#     plot_trajectory(trajectories, 'action', **{'action_names': env.action_names, 'plot_state': np.ones(len(env.action_names))})
""" Script to plot trajectories from a saved file. """
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def plot_trajectory(env, trajectory, field:str, normalized=False, *args, **kwargs):
    fields = ('time', 'state', 'action', 'reward', 'env_info')
    funcs = (lambda:print("Cannot plot time"),
             get_state_df,
             get_action_df,
             get_reward_df,
             get_env_df,
             )
    kwargs['total_observations'] = len(trajectory.reward)

    df = pd.DataFrame()
    for n, f_ in enumerate(fields):
        if field == f_:
            df = funcs[n](env, trajectory, normalized=normalized, **kwargs)
    df.index = [t.item() for t in trajectory.time[:-1]]
    sns.lineplot(df,
                 alpha=kwargs.get('alpha', 1),
                 ax=kwargs.get('ax', None),
                 legend=kwargs.get('legend', False),
                 )
    plot_name = kwargs.get('plot_name', None)
    if plot_name is not None:
        plt.savefig(f'trajectory_plots/{plot_name}.png')
        plt.close()
        return None
    else:
        return trajectory

def get_state_df(env, trajectory, normalized=False, **kwargs):
    plot_mask = kwargs.get("plot_mask" , np.ones(len(env.state_names)).astype(bool))
    columns = [name for name, plot in zip(env.state_names, plot_mask) if plot]
    if normalized:
        data = np.asarray([env.state_scaler.inverse_transform([state])[0] for state in trajectory.state])[:,plot_mask]
    else:
        data = np.asarray([state for state in trajectory.state])[:,plot_mask]
    df = pd.DataFrame(columns=columns, index=range(0, kwargs['total_observations'] + 1), data=data)
    #sns.lineplot(df, alpha=1)
    return df

def get_action_df(env, trajectory, normalized=False, **kwargs):
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
        if normalized:
            data = env.action_scaler.inverse_transform(actions)[:, plot_mask]
        else:
            data = actions[:,plot_mask]
        df.loc[(df.index >= start_row) & (df.index < start_row+T)] = data
        start_row += T
    #sns.lineplot(df, alpha=1)
    return df

def get_reward_df(env, trajectory, **kwargs):
    if kwargs.get('cumulative', False):
        df = pd.DataFrame(data={'Reward': np.cumsum(np.array(trajectory.reward))}, index=range(1, kwargs['total_observations'] + 1))
    else:
        df = pd.DataFrame(data={'Reward': np.array(trajectory.reward)}, index=range(1, kwargs['total_observations'] + 1))
        df["Moving Average (10 days)"] = df['Reward'].rolling(window=10).mean()
    return df
    
def get_env_df(env, trajectory, **kwargs):
    start_row = 1
    df = pd.DataFrame(columns=kwargs['env_info_keys'], index=range(start_row, kwargs['total_observations'] + start_row))
    info = trajectory.env_info[1:]
    for key in kwargs.get('env_info_keys', []):
        data = []
        for x in info[-kwargs['total_observations']:]:
            data += list([np.mean(x[key])])
        data = np.asarray(data)
        if len(data.shape) == 3:
            print("Use action plotting function to plot actions.")
        if len(data.shape) == 2:
            data = np.mean(data, axis=1)
        df.loc[:, key] = data
    #sns.lineplot(df)
    return df

# if __name__ == "__main__":
#     env = make_rfp_env()
#     trajectories = load_trajectories("normalized_test")
#     # plot_trajectory(trajectories, 'env_info', **{'env_info_keys': ['wind_power', 'solar_power']})
#     # plot_trajectory(trajectories, 'reward')
#     # plot_trajectory(trajectories, 'state', **{'state_names': env.state_names, 'plot_state': np.ones(len(env.state_names))})
#     plot_trajectory(trajectories, 'action', **{'action_names': env.action_names, 'plot_state': np.ones(len(env.action_names))})
""" Necessary path addendum if we want to run this script not from the root. 
    To run from root call:
    python -m test_scripts.SCRIPTNAME """
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common_scripts.trajectory_plotlib import plot_trajectory
from common_scripts.utils import load_trajectories, load_stats
from common_scripts.RFP_initialization import create_rfp
from model_scripts.environment import RFPShieldEnv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns

# sns.set_theme("notebook")
# sns.set_context("notebook", font_scale=1.5, rc={"grid":False})
sns.set_theme("paper", font_scale=1.5, style="dark")
plt.rcParams['font.size'] = 16
# set legend fontsize to 14
plt.rcParams['legend.fontsize'] = 18
# set the font weight of the legend to bold
plt.rcParams['legend.title_fontsize'] = 18
# set the font size of the x and y labels to 14
plt.rcParams['axes.labelsize'] = 18
# set the font weight of the x and y labels to bold
plt.rcParams['axes.labelweight'] = 'bold'
# set the font size of the x and y ticks to 12
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
# set the font size of the title to 16
plt.rcParams['axes.titlesize'] = 18
# set the font weight of the title to bold
plt.rcParams['axes.titleweight'] = 'bold'



rfp = create_rfp()

def make_trajectory_figures(env, trajectory, stat, experiment_name, trajectory_index):
    normalized = stat.get("normalized", False)
    experiment_name += "/traj" + str(trajectory_index)
    dn = os.path.dirname("trajectory_plots/" + experiment_name + "/")
    if not os.path.exists(dn):
        os.mkdir(dn)
    trajectory_len = len(trajectory.reward)
    
    #%% Ammonia state plot
    fig, ax = plt.subplots(figsize=(18,12))
    plot_name = experiment_name + '/ammonia_investment_logic'

    nh3_soc = [trajectory.state[t]["state"]["storages"][1] for t in range(trajectory_len+1)]
    nh3_contract = [trajectory.state[t]["state"]["contracts"][1] for t in range(trajectory_len+1)]
    index = [pd.to_datetime(t.item()) for t in trajectory.time]

    ax.plot(index, nh3_soc, color='green', label="Ammonia Storage SOC")
    ax.plot(index, nh3_contract, color='orange', label="Ammonia Contract Status")
    ax.set_ylabel(r"tons NH$_3$")
    ax.set_xlabel("Date")
    ax.set_ylim(bottom=0)
    ax.set_xlim(index[0], index[-1])
    ax.axhline(stat['Ammonia1 volume'], label=r"NH$_3$ contracted volume", color='black', linestyle="-.")
    ax.axhline(stat['Ammonia Storage capacity'], label=r"NH$_3$ storage capacity", color='brown', linestyle="-.", alpha=0.7)

    ax2 = ax.twinx()
    n_days = len(index)-1
    nh3_strike_price = stat.get('ammonia_strike_price',None)[trajectory_index*n_days:(1+trajectory_index)*n_days]
    nh3_electricity_consumption = rfp.get_component("Electrolyzer").parameters.get("electricity_consumption")/rfp.get_component("Haber Bosch Plant").parameters.get("rate") + rfp.get_component("Haber Bosch Plant").parameters.get("electricity_consumption")
    if nh3_strike_price is not None:
        ax2.plot(index[:-1], np.asarray(nh3_strike_price)/nh3_electricity_consumption, color='red', label=r"Internal NH$_3$ strike price [€/MWhe]")
    
    prices_hours = []
    for t in range(1,trajectory_len+1):
        prices_hours += list(trajectory.env_info[t]["electricity_price"])
    prices = []
    for t in range(1,trajectory_len+1):
        prices += [np.mean(prices_hours[24*(t-1):24*t])]
    
    ax2.plot(index[:-1], prices, label="Daily Average Electricity Price", alpha=0.7)
    ax2.set_ylabel(r"€/MWh")
    ax2._remove_legend(ax2.get_legend())
    ax2.set_ylim(bottom=-5)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1+h2,l1+l2, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
    plt.tight_layout()
    plt.savefig(f'trajectory_plots/{plot_name}.png')
    plt.close()

    #%% Hydrogen state plot (End of day SOC)
    fig, ax = plt.subplots(figsize=(18,12))
    plot_name = experiment_name + '/hydrogen_state'
    h2_soc = [trajectory.state[t]["state"]["storages"][0] for t in range(trajectory_len+1)]
    plt.plot(index, h2_soc, color='green', label="Hydrogen Storage (EOD)")
    plt.ylabel(r"tons H$_2$")
    plt.xlabel("Date")
    plt.xlim(index[0],index[-1])
    plt.ylim(0)
    plt.axhline(stat['Hydrogen Storage capacity'], label=r"H$_2$ storage capacity", color='brown', linestyle="-.", alpha=0.7)
    plt.legend()
    plt.savefig(f'trajectory_plots/{plot_name}.png')
    plt.close()

    fig, ax = plt.subplots(figsize=(18,12))
    plot_name = experiment_name + '/hydrogen_state_mean'
    h2soc_hours = []
    for t in range(1,trajectory_len+1):
        h2soc_hours += list(trajectory.env_info[t]["storage_soc"]["Hydrogen Storage"])
    h2soc_mean = []
    for t in range(1,trajectory_len+1):
        h2soc_mean += [np.mean(h2soc_hours[24*(t-1):24*t])]
    plt.plot(index[:-1], h2soc_mean, color='green', label="Hydrogen Storage (Daily Average)")
    plt.ylabel(r"tons H$_2$")
    plt.xlabel("Date")
    plt.xlim(index[0],index[-1])
    plt.ylim(0)
    plt.axhline(stat['Hydrogen Storage capacity'], label=r"H$_2$ storage capacity", color='brown', linestyle="-.", alpha=0.7)
    plt.legend()
    plt.savefig(f'trajectory_plots/{plot_name}.png')
    plt.close()

    # #%% Action plot
    # plot_name = experiment_name + '/action'
    # trajectory = plot_trajectory(env,
    #                                 trajectory,
    #                                 'action',
    #                                 normalized=normalized,
    #                                 )
    # plt.legend()
    # plt.savefig(f'trajectory_plots/{plot_name}.png')
    # plt.close()

    # #%% Grid sale plot
    # trajectory = plot_trajectory(env,
    #                                 trajectory,
    #                                 'action',
    #                                 **{'plot_mask': np.array([1,0,0,0,0]).astype(bool),},
    #                                 normalized=normalized,
    #                                 )
    # plt.legend()
    # plt.savefig(f'trajectory_plots/{plot_name}_grid_sale.png')
    # plt.close()

    # #%% Electrolyzer power plot
    # trajectory = plot_trajectory(env,
    #                                 trajectory,
    #                                 'action',
    #                                 **{'plot_mask': np.array([0,1,0,0,0]).astype(bool),},
    #                                 normalized=normalized,
    #                                 )
    # plt.legend()
    # plt.savefig(f'trajectory_plots/{plot_name}_elec_power.png')
    # plt.close()

    # #%% Pipeline flow plot
    # trajectory = plot_trajectory(env,
    #                                 trajectory,
    #                                 'action',
    #                                 **{'plot_mask': np.array([0,0,1,0,0]).astype(bool),},
    #                                 normalized=normalized,
    #                                 )
    # plt.legend()
    # plt.savefig(f'trajectory_plots/{plot_name}_pipeline_flow.png')
    # plt.close()

    # #%% Balancing power plot
    # plot_name = experiment_name + '/balancing'
    # trajectory = plot_trajectory(env,
    #                                 trajectory,
    #                                 'env_info',
    #                                 **{'env_info_keys': ['balancing'],},
    #                                 normalized=normalized,
    #                                 )
    # plt.legend()
    # plt.savefig(f'trajectory_plots/{plot_name}.png')
    # plt.close()

    # #%% Hydrogen SOC plot (Daily mean)
    # plot_name = experiment_name + '/soc_h2'
    # trajectory = plot_trajectory(env,
    #                                 trajectory,
    #                                 'env_info',
    #                                 **{'env_info_keys': ['soc_h2'],},
    #                                 normalized=normalized,
    #                                 )
    # plt.legend()
    # plt.savefig(f'trajectory_plots/{plot_name}.png')
    # plt.close()

    # #%% Technical feasibility violation penalty plot
    # plot_name = experiment_name + '/technical_violation_cost'
    # trajectory = plot_trajectory(env,
    #                                 trajectory,
    #                                 'env_info',
    #                                 **{'env_info_keys': ['technical_violation_cost'],},
    #                                 normalized=normalized,
    #                                 )
    # plt.legend()
    # plt.savefig(f'trajectory_plots/{plot_name}.png')
    # plt.close()

    #%% Average daily electricity price plot
    fig, ax = plt.subplots(figsize=(18,12))
    plot_name = experiment_name + '/electricity_price'
    plt.plot(index[:-1], prices, label="Daily Average Electricity Price", alpha=0.7)
    plt.xlim(index[0], index[-1])
    plt.ylabel("€/MWh")
    plt.xlabel("Date")
    plt.legend()
    plt.savefig(f'trajectory_plots/{plot_name}.png')
    plt.close()

    #%% Average daily electricity price plot
    fig, ax = plt.subplots(figsize=(18,12))
    plot_name = experiment_name + '/reward'
    reward = [r.item() for r in trajectory.reward]
    cum_reward = np.cumsum(reward)
    plt.xlim(index[0], index[-1])
    plt.ylabel("€ (million)")
    plt.xlabel("Date")
    plt.plot(index[:-1],cum_reward/1e6, label="Cumulative Reward")
    plt.legend()
    plt.savefig(f'trajectory_plots/{plot_name}.png')
    plt.close()

if __name__ == '__main__':
    """ Change the experiment name to the one we want to visualize: """
    experiment_names = ("test_BiddingCurveAgent1_D1_ph_96_spot_True",
                        "test_BiddingCurveAgent1_D2_ph_96_spot_True",
                        "test_BiddingCurveAgent1_D3_ph_96_spot_True",
                        "test_DeterministicHA_hourly_target_ph_96_spot_True",
                        "test_DeterministicHA_production_value_ph_96_spot_True",
                        "test_StochasticHA5_production_value_ph_96_spot_True",
                        "test_RecourseAgent1_production_value_ph_96_spot_True",
                        "test_RecourseAgent5_production_value_ph_96_spot_True",
                        "test_StrikePriceBiddingAgent1_SP1_production_value_ph_96_spot_True"
                        )
    planning_horizon = 4 * 24
    decision_horizon = 24
    env = RFPShieldEnv(rfp=rfp, decision_horizon=decision_horizon, planning_horizon=planning_horizon)
    for experiment_name in experiment_names:
        """ Plotting the first trajectory of the experiment """
        print(experiment_name)
        trajectories = load_trajectories(experiment_name)
        stats = load_stats(experiment_name, csv_version=False)
        # experiment_name += "_new"

        dn = os.path.dirname("trajectory_plots/" + experiment_name + "/")
        if not os.path.exists(dn):
            os.mkdir(dn)

        #%% Trajectory summary:
        print("Experiment: ", experiment_name)
        for ix in range(min(10,len(trajectories))):
            make_trajectory_figures(env, trajectory=trajectories[ix], stat = stats[ix], experiment_name=experiment_name, trajectory_index=ix)

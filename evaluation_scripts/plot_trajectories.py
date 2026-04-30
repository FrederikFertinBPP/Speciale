""" Necessary path addendum if we want to run this script not from the root. 
    To run from root call:
    python -m test_scripts.SCRIPTNAME """
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common_scripts.utils import load_trajectories, load_stats, set_plotting_style
from common_scripts.RFP_initialization import create_rfp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
set_plotting_style()

def make_trajectory_figures(rfp, trajectory, stat, experiment_name, trajectory_index):
    experiment_name += "/traj" + str(trajectory_index)
    dn = os.path.dirname("trajectory_plots/" + experiment_name + "/")
    if not os.path.exists(dn):
        os.mkdir(dn)
    trajectory_len = len(trajectory.reward)

    nh3_electricity_consumption = rfp.get_component("Electrolyzer").parameters.get("electricity_consumption")/rfp.get_component("Haber Bosch Plant").parameters.get("rate") + rfp.get_component("Haber Bosch Plant").parameters.get("electricity_consumption")
    
    #%% Ammonia state plot
    fig, ax = plt.subplots(figsize=(14,7))
    plot_name = experiment_name + '/ammonia_investment_logic'

    nh3_soc = [trajectory.state[t]["state"]["storages"][1] for t in range(trajectory_len+1)]
    nh3_contract = [trajectory.state[t]["state"]["contracts"][1] for t in range(trajectory_len+1)]
    index = [pd.to_datetime(t.item()) for t in trajectory.time]

    ax.plot(index, nh3_soc, color='green', label=r"NH$_3$ storage SOC", lw=3)
    ax.plot(index, nh3_contract, color='red', label=r"NH$_3$ contract status", lw=3)
    ax.set_ylabel(r"tons NH$_3$")
    ax.set_xlabel("Date")
    ax.axhline(stat['Ammonia Storage capacity'], label=r"NH$_3$ storage capacity", color='darkgreen', linestyle="--", lw=3)
    ax.axhline(stat['Ammonia1 volume'], label=r"NH$_3$ contracted volume", color='darkred', linestyle="--", lw=3)
    ax.set_ylim(bottom=0)
    ax.set_xlim(index[0], index[-1])

    ax2 = ax.twinx()
    n_days = len(index)-1
    nh3_strike_price = stat.get('ammonia_strike_price',None)
    
    prices_hours = []
    for t in range(1,trajectory_len+1):
        prices_hours += list(trajectory.env_info[t]["electricity_price"])
    prices = []
    for t in range(1,trajectory_len+1):
        prices += [np.mean(prices_hours[24*(t-1):24*t])]
    # Get moving average of prices
    movavg_window = 7
    prices = pd.Series(prices).rolling(window=movavg_window, min_periods=1).mean().tolist()

    ax2.plot(index[:-1], prices, label="7-day moving-average electricity price", alpha=0.3, color='black', lw=3)
    if nh3_strike_price is not None:
        ax2.plot(index[:-1], np.asarray(nh3_strike_price[trajectory_index*n_days:(1+trajectory_index)*n_days])/nh3_electricity_consumption,
                 color='purple', linestyle="-.", label=r"Internal NH$_3$ strike price [€/MWhe]", lw=3)
    ax2.set_ylabel(r"€/MWh")
    ax2._remove_legend(ax2.get_legend())
    ax2.set_ylim(bottom=-5)
    ax2.grid(False)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    handles = h1[:2] + h2[:1] + h1[2:] + h2[1:]
    labels = l1[:2] + l2[:1] + l1[2:] + l2[1:]
    ax.legend(handles,labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
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

    #%% Average daily electricity price plot
    fig, ax = plt.subplots(figsize=(18,12))
    plot_name = experiment_name + '/electricity_price'
    plt.plot(index[:-1], prices, label="Daily Average Electricity Price", alpha=0.7, lw=1)
    if nh3_strike_price is not None:
        plt.plot(index[:-1], np.asarray(nh3_strike_price[trajectory_index*n_days:(1+trajectory_index)*n_days])/nh3_electricity_consumption,
                    color='purple', linestyle="-.", label=r"Internal NH$_3$ strike price [€/MWhe]", lw=1)
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

    import seaborn as sns
    fig,ax = plt.subplots()
    plot_name = experiment_name + '/electrolyzer_util_histogram'
    e_cap = stat['components']['Electrolyzer'].parameters['capacity']
    e_power = []
    for t in range(len(trajectory.reward)):
        e_power += list(trajectory.env_info[t+1]['link_productions']['Electrolyzer'])
    sns.histplot(data=np.asarray(e_power)/e_cap*100,bins=50)
    plt.xlim(0,100)
    plt.ylabel("Occurrences")
    plt.xlabel("Utilization (%)")
    plt.savefig(f'trajectory_plots/{plot_name}.png')
    plt.close()

    fig,ax = plt.subplots()
    plot_name = experiment_name + '/hb_util_histogram'
    hb_cap = stat['components']['Haber Bosch Plant'].parameters['capacity']
    hb_power = []
    for t in range(len(trajectory.reward)):
        hb_power += list(trajectory.env_info[t+1]['link_productions']['Haber Bosch Plant'])
    sns.histplot(data=np.asarray(hb_power)/(hb_cap)*100,bins=50)
    plt.xlim(0,100)
    plt.ylabel("Occurrences")
    plt.xlabel("Utilization (%)")
    plt.savefig(f'trajectory_plots/{plot_name}.png')
    plt.close()

    fig,ax = plt.subplots()
    plot_name = experiment_name + '/hb_rolling'
    rolling_nh3_production = pd.Series(hb_power).rolling(window=168, min_periods=1).mean().tolist()
    plt.plot(rolling_nh3_production)
    plt.ylabel("t NH3/h")
    plt.xlabel("Hour of year")
    plt.savefig(f'trajectory_plots/{plot_name}.png')
    plt.close()
    print(f"Saved trajectory plots to trajectory_plots/{experiment_name}/")

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
                        "test_StrikePriceBiddingAgent1_SP1_production_value_ph_96_spot_True",
                        )
    experiment_names = ("test_DeterministicHA_hourly_target_ph_96_spot_True",)
    experiment_names = ("storage_cost_test_DeterministicHA_production_value_ph_96_spot_True_small",)
    experiment_names = ("backcasting_DeterministicHA_production_value_ph_96_spot_True_small",
                        "backcasting_AggregateFullHorizonAgent_ph_96_spot_True_small",
                        "backcasting_prophet_DeterministicHA_production_value_ph_96_spot_True_small",
                        "backcasting_persistence_DeterministicHA_production_value_ph_96_spot_True_small")
    # experiment_names = ("test_DeterministicHA_production_value_ph_96_spot_True_small",)
    # experiment_names = ("minload_pwl_DeterministicHA_production_value_ph_96_spot_True_small",
    #                     "minload_pwl_ramp_DeterministicHA_production_value_ph_96_spot_True_small",
    #                     "planningsensitivity_DeterministicHA_production_value_ph_96_spot_True_small",
    #                     "test_AggregateFullHorizonAgent_ph_96_spot_True_small")
    rfp = create_rfp(scenario_name="default")
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
            make_trajectory_figures(rfp=rfp, trajectory=trajectories[ix], stat = stats[ix], experiment_name=experiment_name, trajectory_index=ix)

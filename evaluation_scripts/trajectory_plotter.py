""" Necessary path addendum if we want to run this script not from the root. 
    To run from root call:
    python -m test_scripts.SCRIPTNAME """
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common_scripts.trajectory_plotlib import plot_trajectory
from common_scripts.utils import load_trajectories, load_stats
from common_scripts.RFP_initialization import create_rfp
from model_scripts_old.RFP_operational_environment import RFPOperationalEnv
import numpy as np
import matplotlib.pyplot as plt
import os

def make_trajectory_figures(env, trajectory, stats, experiment_name, trajectory_index):
    normalized = stats.get("normalized", False)
    nh3_ammonia1_target = stats['contracted_yearly_targets']['Ammonia1'].parameters.get('volume')
    experiment_name += "/traj" + str(trajectory_index)
    dn = os.path.dirname("trajectory_plots/" + experiment_name + "/")
    if not os.path.exists(dn):
        os.mkdir(dn)
    #%% Ammonia state plot
    fig, ax = plt.subplots(figsize=(18,12))
    plot_name = experiment_name + '/ammonia_investment_logic'
    trajectory = plot_trajectory(env,
                                    trajectory,
                                    'state',
                                    **{'plot_mask': np.array([0,1,1,1]).astype(bool),
                                    'ax':ax,
                                    'legend':False},
                                    normalized=normalized,
                                    )
    ax.set_ylabel(r"tons NH$_3$")
    ax.set_xlabel("Days")
    ax.set_xlim(0, len(trajectory.reward))
    ax.axhline(nh3_ammonia1_target, label=r"NH$_3$ contract quantity", color='black', linestyle="-.")
    ax.axhline(stats['ammonia_storage_capacity'], label=r"NH$_3$ storage capacity", color='brown', linestyle="-.", alpha=0.7)

    ax2 = ax.twinx()
    nh3_strike_price = stats.get('ammonia_strike_price',None)
    if nh3_strike_price is not None:
        ax2.plot(nh3_strike_price, color='red', label=r"Estimated NH$_3$ strike price")
    trajectory = plot_trajectory(env,
                                    trajectory,
                                    'env_info',
                                    **{'env_info_keys': ['electricity_price'],'ax':ax,'alpha':0.5, 'legend':False},
                                    normalized=normalized,
                                    )
    ax2.set_ylabel(r"€/MWh")
    ax2._remove_legend(ax2.get_legend())

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1+h2,l1+l2, loc='upper left')
    plt.savefig(f'trajectory_plots/{plot_name}.png')
    plt.close()

    #%% Ammonia state plot 
    plot_name = experiment_name + '/ammonia_state'
    trajectory = plot_trajectory(env,
                                    trajectory,
                                    'state',
                                    **{'plot_mask': np.array([0,1,1,1]).astype(bool),},
                                    normalized=normalized,
                                    )
    plt.ylabel(r"tons NH$_3$")
    plt.xlabel("Days")
    plt.xlim(0, len(trajectory.reward))
    plt.axhline(nh3_ammonia1_target, label=r"NH$_3$ contract quantity", color='black', linestyle="-.")
    plt.axhline(stats['ammonia_storage_capacity'], label=r"NH$_3$ storage capacity", color='brown', linestyle="-.", alpha=0.7)
    # plt.axhline(env.state_space.high[3], label=r"Monthly max. NH$_3$ spot sale", color='black', linestyle=":")
    plt.legend()
    plt.savefig(f'trajectory_plots/{plot_name}.png')
    plt.close()


    #%% Hydrogen state plot (End of day SOC)
    plot_name = experiment_name + '/hydrogen_state'
    trajectory = plot_trajectory(env,
                                    trajectory,
                                    'state',
                                    **{'plot_mask': np.array([1,0,0,0]).astype(bool),},
                                    normalized=normalized,
                                    )
    plt.ylabel(r"tons H$_2$")
    plt.xlabel("Days")
    plt.xlim(0, len(trajectory.reward))
    plt.axhline(stats['hydrogen_storage_capacity'], label=r"H$_2$ storage capacity", color='brown', linestyle="-.", alpha=0.7)
    plt.legend()
    plt.savefig(f'trajectory_plots/{plot_name}.png')
    plt.close()

    #%% Action plot
    plot_name = experiment_name + '/action'
    trajectory = plot_trajectory(env,
                                    trajectory,
                                    'action',
                                    normalized=normalized,
                                    )
    plt.legend()
    plt.savefig(f'trajectory_plots/{plot_name}.png')
    plt.close()

    #%% Grid sale plot
    trajectory = plot_trajectory(env,
                                    trajectory,
                                    'action',
                                    **{'plot_mask': np.array([1,0,0,0,0]).astype(bool),},
                                    normalized=normalized,
                                    )
    plt.legend()
    plt.savefig(f'trajectory_plots/{plot_name}_grid_sale.png')
    plt.close()

    #%% Electrolyzer power plot
    trajectory = plot_trajectory(env,
                                    trajectory,
                                    'action',
                                    **{'plot_mask': np.array([0,1,0,0,0]).astype(bool),},
                                    normalized=normalized,
                                    )
    plt.legend()
    plt.savefig(f'trajectory_plots/{plot_name}_elec_power.png')
    plt.close()

    #%% Pipeline flow plot
    trajectory = plot_trajectory(env,
                                    trajectory,
                                    'action',
                                    **{'plot_mask': np.array([0,0,1,0,0]).astype(bool),},
                                    normalized=normalized,
                                    )
    plt.legend()
    plt.savefig(f'trajectory_plots/{plot_name}_pipeline_flow.png')
    plt.close()

    #%% Balancing power plot
    plot_name = experiment_name + '/balancing'
    trajectory = plot_trajectory(env,
                                    trajectory,
                                    'env_info',
                                    **{'env_info_keys': ['balancing'],},
                                    normalized=normalized,
                                    )
    plt.legend()
    plt.savefig(f'trajectory_plots/{plot_name}.png')
    plt.close()

    #%% Hydrogen SOC plot (Daily mean)
    plot_name = experiment_name + '/soc_h2'
    trajectory = plot_trajectory(env,
                                    trajectory,
                                    'env_info',
                                    **{'env_info_keys': ['soc_h2'],},
                                    normalized=normalized,
                                    )
    plt.legend()
    plt.savefig(f'trajectory_plots/{plot_name}.png')
    plt.close()

    #%% Technical feasibility violation penalty plot
    plot_name = experiment_name + '/technical_violation_cost'
    trajectory = plot_trajectory(env,
                                    trajectory,
                                    'env_info',
                                    **{'env_info_keys': ['technical_violation_cost'],},
                                    normalized=normalized,
                                    )
    plt.legend()
    plt.savefig(f'trajectory_plots/{plot_name}.png')
    plt.close()

    #%% Average daily electricity price plot
    plot_name = experiment_name + '/electricity_price'
    trajectory = plot_trajectory(env,
                                    trajectory,
                                    'env_info',
                                    **{'env_info_keys': ['electricity_price'],},
                                    normalized=normalized,
                                    )
    plt.legend()
    plt.savefig(f'trajectory_plots/{plot_name}.png')
    plt.close()

    #%% Average daily electricity price plot
    plot_name = experiment_name + '/reward'
    trajectory = plot_trajectory(env,
                                    trajectory,
                                    'reward',
                                    **{"cumulative":False, 'legend':True},
                                    normalized=normalized,
                                    )
    plt.savefig(f'trajectory_plots/{plot_name}.png')
    plt.close()

if __name__ == '__main__':
    """ Change the experiment name to the one we want to visualize: """
    # experiment_name = "test3_strike_price"
    experiment_name = "test3_planning_target"
    experiment_name = "testRandom"
    experiment_name = "testContextAwareRL"
    experiment_name = "testStateAwareSteeringDDPG"

    """ Plotting the first trajectory of the experiment """
    rfp = create_rfp()
    planning_horizon = 4 * 24
    decision_horizon = 24
    env = RFPOperationalEnv(rfp=rfp, decision_horizon=decision_horizon, planning_horizon=planning_horizon)

    trajectories = load_trajectories(experiment_name)
    stats = load_stats(experiment_name, csv_version=False)
    # experiment_name += "_new"

    dn = os.path.dirname("trajectory_plots/" + experiment_name + "/")
    if not os.path.exists(dn):
        os.mkdir(dn)

    #%% Trajectory summary:
    print("Experiment: ", experiment_name)
    for ix in range(len(trajectories)):
        make_trajectory_figures(env, trajectory=trajectories[ix], stats = stats[ix], experiment_name=experiment_name, trajectory_index=ix)
        realized_action = [x["adjusted_action"] for x in trajectories[ix].env_info[1:]]
        nh3_production = np.asarray([x[:,3] for x in realized_action]).reshape(-1)

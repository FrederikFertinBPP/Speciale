""" Necessary path addendum if we want to run this script not from the root. 
    To run from root call:
    python -m test_scripts.SCRIPTNAME """
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common_scripts.utils import load_trajectories, load_stats
from common_scripts.RFP_initialization import create_rfp
from model_scripts_old.RFP_operational_environment import SpotBuyRFPEnv
from trajectory_plotter import make_trajectory_figures
import os

if __name__ == '__main__':
    """ Change the experiment name to the one we want to visualize: """
    experiment_name = "testspotbuy_planning_target"
    # experiment_name = "testspotbuy_strike_price"
    experiment_name = "testSpotbuy_strike_price"
    experiment_name = "testRandomSpotBuy"
    experiment_name = "testDeterministicSpotbuy_strike_price"
    # experiment_name = "testStochasticSpotbuy_strike_price"
    experiment_name = "testContextAwareRLSpotbuy"


    """ Plotting the first trajectory of the experiment """
    rfp = create_rfp()
    planning_horizon = 4 * 24
    decision_horizon = 24
    env = SpotBuyRFPEnv(rfp=rfp, decision_horizon=decision_horizon, planning_horizon=planning_horizon)

    trajectories = load_trajectories(experiment_name)
    stats = load_stats(experiment_name, csv_version=False)
    # experiment_name += "_new"

    dn = os.path.dirname("trajectory_plots/" + experiment_name + "/")
    if not os.path.exists(dn):
        os.mkdir(dn)

    #%% Trajectory summary:
    print("Experiment: ", experiment_name)
    for ix in range(len(trajectories)):
        make_trajectory_figures(env, trajectory=trajectories[ix], stats=stats[ix], experiment_name=experiment_name, trajectory_index=ix)

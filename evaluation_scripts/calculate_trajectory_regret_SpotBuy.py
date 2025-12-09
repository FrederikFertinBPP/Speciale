""" Necessary path addendum if we want to run this script not from the root. 
    To run from root call:
    python -m test_scripts.SCRIPTNAME """
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common_scripts.utils import load_trajectories, load_stats
from model_scripts_old.lp_deterministic import SpotBuyHourlyDeterministicLPModel
from calculate_trajectory_regret import print_trajectory_summary, rfp
from model_scripts_old.RFP_operational_environment import SpotBuyRFPEnv
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    """ Change the experiment name to the one we want to evaluate: """
    experiment_name = "testSpotbuy_strike_price"
    experiment_name = "testRandomSpotBuy"
    # experiment_name = "testspotbuy_planning_target"
    experiment_name = "testStochasticSpotbuy_strike_price"
    experiment_name = "testDeterministicSpotbuy_strike_price"
    # experiment_name = "testConstantSpotBuy"
    experiment_name = "testContextAwareRLSpotbuy"
    experiment_name = "testStochasticSpotbuy_strike_price"

    #### We assess the regret of the model:
    # This is the difference in profits between the actions chosen by the agent and the optimal actions
    # chosen by an oracle: A perfect foresight model for the full year.
    # Just load the realized wind, solar, and prices (can be found in trajectory env info)
    # and solve an LP for the full year.

    #%% Trajectory stats:
    trajectories = load_trajectories(experiment_name)
    stats = load_stats(experiment_name, csv_version=False)
    print("\n----------------------------------\n")
    print("Experiment: ", experiment_name)

    profit_percentages = []
    opt_models = []
    for ix in range(len(trajectories)):
        pp, opt_model = print_trajectory_summary(trajectory=trajectories[ix], stats=stats[ix], model_class=SpotBuyHourlyDeterministicLPModel)
        profit_percentages.append(pp)
        opt_models.append(opt_model)
    print("Done")
    plt.hist(np.asarray(profit_percentages), label="Obtained profits")
    plt.axvline(np.mean(profit_percentages), label="Average obtained profit", lw=5, color='black')
    plt.xlabel(f"% of optimal profits")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(f"documentation/profit_dists/{experiment_name}.png")
    plt.close()
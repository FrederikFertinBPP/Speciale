""" Necessary path addendum if we want to run this script not from the root. 
    To run from root call:
    python -m test_scripts.SCRIPTNAME """
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common_scripts import train
from model_scripts.environment import RFPRecourseEnv, get_env
from model_scripts.agent_hierarchical_heuristic import BiddingCurveAgent

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme("notebook", font_scale=1.5, style="darkgrid")
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

def main():
    n_episodes = 50
    solver = 'gurobi'
    planning_horizon = 4*24
    allow_spot_buy = True
    n_scenarios = 1
    guideline = "production_value"

    env = get_env(RFPRecourseEnv, allow_spot_buy=allow_spot_buy, balancing_market=True, verbose=True ,load_data=True)
    
    agent = BiddingCurveAgent(env=env, solver=solver, documentation=False,
                              guideline=guideline, planning_horizon=planning_horizon, n_scenarios=n_scenarios,
                              mode="eval", no_train=True,
                              n_price_domains=2, domain_prices=[80],
                            )
    training_experiment = "_".join(["train", str(agent), "spot", str(allow_spot_buy)])
    agent.load(os.getcwd() + f"/models/rl_models/{training_experiment}")
    
    experiment_name = "_".join(["test", str(agent), "ph", str(planning_horizon), "spot", str(allow_spot_buy)])
    print("Start experiment: ", experiment_name)
    stats, trajectories = train(env, agent, experiment_name=experiment_name, num_episodes=n_episodes, save_every=10)
    agent.close()

    print("Experiment done")


import cProfile
if __name__ == '__main__':
    cProfile.run("main()", "run_profiles/run_TrainDecisionRule.prof")
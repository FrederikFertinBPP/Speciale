""" Necessary path addendum if we want to run this script not from the root. 
    To run from root call:
    python -m test_scripts.SCRIPTNAME """
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common_scripts import train
from model_scripts.environment import RFPRecourseEnv, get_env
from model_scripts.agent_hierarchical_heuristic import BiddingCurveAgent

def main():
    n_episodes = 1
    solver = 'gurobi'
    planning_horizon = 4*24
    allow_spot_buy = True
    n_scenarios = 1
    guideline = "production_value"

    env_config = {"allow_spot_buy": allow_spot_buy, "balancing_market": True, "verbose": True, "load_data": True, "inflexible": False}
    env = get_env(RFPRecourseEnv, env_config=env_config, layout_file="rfp_layout - resized.xlsx")
    
    agent = BiddingCurveAgent(env=env, solver=solver, documentation=False,
                              guideline=guideline, planning_horizon=planning_horizon, n_scenarios=n_scenarios,
                              mode="eval", no_train=True,
                              n_price_domains=3, domain_prices=[80, 80*1.4],
                            )
    training_experiment = "_".join(["train", str(agent), "spot", str(allow_spot_buy)])
    agent.load(os.getcwd() + f"/models/rl_models/{training_experiment}")
    
    experiment_name = "_".join(["test", str(agent), "ph", str(planning_horizon), "spot", str(allow_spot_buy)])
    print("Start experiment: ", experiment_name)
    # stats, trajectories = train(env, agent, experiment_name=experiment_name, num_episodes=n_episodes, save_every=10)
    stats, trajectories = train(env, agent, num_episodes=1, verbose=True)
    agent.close()

    print("Experiment done")


import cProfile
if __name__ == '__main__':
    cProfile.run("main()", "run_profiles/run_DRAgent_3.prof")
    # Example of how to read the profile results:
    import pstats
    prof = pstats.Stats("run_profiles/run_DRAgent_3.prof")
    prof.strip_dirs().sort_stats("cumtime").print_stats(10)
    # cProfile.py -- Profile Python programs
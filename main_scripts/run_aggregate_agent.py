""" Necessary path addendum if we want to run this script not from the root. 
    To run from root call in terminal: python -m test_scripts.SCRIPTNAME
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common_scripts import train # stats, trajectories = train(env, agent, num_episodes=n_episodes, verbose=True, seed=42)
from model_scripts.environment import RFPModelActionsEnv, get_env
from model_scripts.agent_hierarchical_heuristic import AggregateFullHorizonAgent

def main():
    rfp_case = "small"
    planning_horizon = 96
    allow_spot_buy = True
    solver = 'gurobi'
    
    n_episodes = 1
    
    env_config = {"allow_spot_buy": allow_spot_buy, "balancing_market": False, "verbose": True, "load_data": True, "inflexible": True}
    env = get_env(RFPModelActionsEnv, env_config=env_config, layout_file="rfp_layout - resized.xlsx")
    
    agent = AggregateFullHorizonAgent(env=env, solver=solver, planning_horizon=planning_horizon, documentation=False)
    
    experiment_name = "_".join(["test", str(agent), "ph", str(planning_horizon), "spot", str(allow_spot_buy), rfp_case])
    print("Start experiment: ", experiment_name)
    stats, trajectories = train(env, agent, num_episodes=1, verbose=True, experiment_name=experiment_name)
    agent.close()
    print("Experiment done")


import cProfile
if __name__ == '__main__':
    cProfile.run("main()", "run_profiles/run_Aggregate96.prof")
    # Example of how to read the profile results:
    import pstats
    prof = pstats.Stats("run_profiles/run_Aggregate96.prof")
    prof.strip_dirs().sort_stats("cumtime").print_stats(10)
    # cProfile.py -- Profile Python programs

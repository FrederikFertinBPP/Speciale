""" Necessary path addendum if we want to run this script not from the root. 
    To run from root call in terminal: python -m test_scripts.SCRIPTNAME
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common_scripts import train # stats, trajectories = train(env, agent, num_episodes=n_episodes, verbose=True, seed=42)
from model_scripts.environment import RFPRecourseEnv, get_env
from model_scripts.agent_hierarchical_heuristic import RecourseAgent

def main():
    allow_spot_buy = True
    planning_horizon = 4 * 24
    n_scenarios = 1 # 1 is deterministic, above one is number of independent scenarios considered by stochastic model.
    solver = 'gurobi'

    """ Set experiment name """
    guideline = "production_value" # Reward production of ammonia based on estimating internal value of ammonia.
    
    n_episodes = 1
    
    env = get_env(RFPRecourseEnv, allow_spot_buy=allow_spot_buy, balancing_market=True, verbose=True, load_data=True)

    agent = RecourseAgent(env=env, solver=solver, planning_horizon=planning_horizon, guideline=guideline, n_scenarios=n_scenarios)

    experiment_name = "_".join(["test", str(agent), guideline, "ph", str(planning_horizon), "spot", str(allow_spot_buy)])
    print("Start experiment: ", experiment_name)
    
    # stats, trajectories = train(env, agent, experiment_name=experiment_name, num_episodes=n_episodes,
    #                             verbose=True, continue_trajectories_and_stats=True, starting_episode=30, save_every=5)
    stats, trajectories = train(env, agent, num_episodes=1, verbose=True)
    agent.close()
    print("Experiment done")


import cProfile
if __name__ == '__main__':
    cProfile.run("main()", "run_profiles/run_RecourseAgent.prof")
    # Example of how to read the profile results:
    import pstats
    prof = pstats.Stats("run_profiles/run_RecourseAgent.prof")
    prof.strip_dirs().sort_stats("cumtime").print_stats(10)
    # cProfile.py -- Profile Python programs
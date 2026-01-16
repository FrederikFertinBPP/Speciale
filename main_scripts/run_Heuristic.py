""" Necessary path addendum if we want to run this script not from the root. 
    To run from root call in terminal: python -m test_scripts.SCRIPTNAME
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common_scripts import train # stats, trajectories = train(env, agent, num_episodes=n_episodes, verbose=True, seed=42)
from model_scripts.environment import RFPEnv, get_env
from model_scripts.agent_hierarchical_heuristic import DeterministicHA

def main():
    planning_horizon = 96
    allow_spot_buy = True
    solver = 'gurobi'

    # guideline = "hourly_target" # Promote a fixed production flow for all hours.
    guideline = "production_value" # Reward production of ammonia based on estimating internal value of ammonia.
    
    n_episodes = 1
    
    env = get_env(RFPEnv, allow_spot_buy=allow_spot_buy, balancing_market=False, verbose=True, load_data=True)
    
    agent = DeterministicHA(env=env, solver=solver, planning_horizon=planning_horizon, guideline=guideline, documentation=True)
    
    experiment_name = "_".join(["planningsensitivity", str(agent), guideline, "ph", str(planning_horizon), "spot", str(allow_spot_buy)])
    print("Start experiment: ", experiment_name)
    stats, trajectories = train(env, agent, num_episodes=1, verbose=True)
    agent.close()
    print("Experiment done")


import cProfile
if __name__ == '__main__':
    cProfile.run("main()", "run_profiles/run_DeterministicSpotBuy.prof")
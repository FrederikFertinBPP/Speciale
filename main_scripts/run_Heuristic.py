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
    allow_spot_buy = True
    planning_horizon = 4 * 24
    solver = 'gurobi'

    # guideline = "hourly_target" # Promote a fixed production flow for all hours.
    guideline = "production_value" # Reward production of ammonia based on estimating internal value of ammonia.
    
    n_episodes = 50
    seeds = [x for x in range(n_episodes)]
    
    env = get_env(RFPEnv, allow_spot_buy=allow_spot_buy, balancing_market=False, verbose=True, load_data=True)
    
    agent = DeterministicHA(env=env, solver=solver, planning_horizon=planning_horizon, guideline=guideline)
    
    experiment_name = "_".join(["test", str(agent), guideline, "ph", str(planning_horizon), "spot", str(allow_spot_buy)])
    print("Start experiment: ", experiment_name)
    stats, trajectories = train(env, agent, experiment_name=experiment_name, num_episodes=n_episodes,
                                verbose=True, seed=seeds, save_every=10,
                                )
    agent.close()
    print("Experiment done")


import cProfile
if __name__ == '__main__':
    cProfile.run("main()", "run_profiles/run_DeterministicSpotBuy.prof")
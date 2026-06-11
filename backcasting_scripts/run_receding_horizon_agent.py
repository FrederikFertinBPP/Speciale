""" Necessary path addendum if we want to run this script not from the root. 
    To run from root call in terminal: python -m test_scripts.SCRIPTNAME
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common_scripts import train # stats, trajectories = train(env, agent, num_episodes=n_episodes, verbose=True, seed=42)
from model_scripts.environment import RFPBackcastEnv, get_env
from model_scripts.agent_hierarchical_heuristic import RecedingHorizonAgent
import pandas as pd

def main():
    rfp_case = "out_of_sample"
    planning_horizon = 96
    allow_spot_buy = True
    solver = 'gurobi'
    
    n_episodes = 1
    
    env_config = {"allow_spot_buy": allow_spot_buy, "balancing_market": False, "verbose": True, "load_data": True, "inflexible": True}

    forecaster_type = "SOTA_combined" # SOTA, SOTA1year, SOTA1yearperfect24hours, SOTA1yearperfect48hours.

    scenarios = [70, 80, 90, 100, 110, 120]
    for scenario in scenarios:
        env = get_env(RFPBackcastEnv, env_config=env_config, layout_file="article.xlsx", use_optimized_capacities=True, scenario_name=str(scenario))
        
        agent = RecedingHorizonAgent(env=env, solver=solver, planning_horizon=planning_horizon, documentation=False)
        
        experiment_name = "_".join([forecaster_type, str(agent), "ph", str(planning_horizon), "spot", str(allow_spot_buy), rfp_case, str(scenario)])
        print("Start experiment: ", experiment_name)
        options = {"episode_start": pd.Timestamp("20250101 000000"), "forecaster_type": forecaster_type}
        stats, trajectories = train(env, agent, num_episodes=1, verbose=True, experiment_name=experiment_name, options=options)
        agent.close()
        print(f"Scenario {scenario}kt NH3 done")
    print("Experiment done")


import cProfile
if __name__ == '__main__':
    cProfile.run("main()", "run_profiles/run_receding.prof")
    # Example of how to read the profile results:
    import pstats
    prof = pstats.Stats("run_profiles/run_receding.prof")
    prof.strip_dirs().sort_stats("cumtime").print_stats(10)
    # cProfile.py -- Profile Python programs

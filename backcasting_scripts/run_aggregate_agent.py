""" Necessary path addendum if we want to run this script not from the root. 
    To run from root call in terminal: python -m test_scripts.SCRIPTNAME
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common_scripts import train # stats, trajectories = train(env, agent, num_episodes=n_episodes, verbose=True, seed=42)
from model_scripts.environment import RFPBackcastEnv, get_env
from model_scripts.agent_hierarchical_heuristic import AggregateFullHorizonAgent
import pandas as pd
import numpy as np

def main():
    rfp_case = "out_of_sample"
    planning_horizon = 96
    allow_spot_buy = True
    solver = 'gurobi'
    
    n_episodes = 1
    
    env_config = {"allow_spot_buy": allow_spot_buy, "balancing_market": False, "verbose": True, "load_data": True, "inflexible": True}
    train_periods = np.concatenate([np.linspace(1,4,19), np.linspace(4.5,10,12)])
    forecaster_types = [f"SOTA{str(round(float(train_period),2)).replace(".","_")}year" for train_period in train_periods]
    forecaster_types = ["SOTA_combined"]
    n_episodes = 1
    for forecaster_type in forecaster_types:
        # scenarios = ["default",]
        scenarios = [70, 80, 90, 100, 110, 120]
        for scenario in scenarios:
            env = get_env(RFPBackcastEnv, env_config=env_config, layout_file="article.xlsx", use_optimized_capacities=True, scenario_name=str(scenario))
            
            agent = AggregateFullHorizonAgent(env=env, solver=solver, planning_horizon=planning_horizon, documentation=False)
            
            experiment_name = "_".join([forecaster_type, str(agent), "ph", str(planning_horizon), "spot", str(allow_spot_buy), rfp_case, str(scenario)])
            print("Start experiment: ", experiment_name)
            options = {"episode_start": pd.Timestamp("20250101 000000"), "forecaster_type": forecaster_type}
            stats, trajectories = train(env, agent, num_episodes=1, verbose=True, experiment_name=experiment_name, options=options)
            agent.close()
            print(f"Scenario {scenario}kt NH3 done")


import cProfile
if __name__ == '__main__':
    cProfile.run("main()", "run_profiles/run_Aggregate96.prof")
    # Example of how to read the profile results:
    import pstats
    prof = pstats.Stats("run_profiles/run_Aggregate96.prof")
    prof.strip_dirs().sort_stats("cumtime").print_stats(10)
    # cProfile.py -- Profile Python programs

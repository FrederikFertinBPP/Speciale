""" Necessary path addendum if we want to run this script not from the root. 
    To run from root call in terminal: python -m test_scripts.SCRIPTNAME
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common_scripts import train # stats, trajectories = train(env, agent, num_episodes=n_episodes, verbose=True, seed=42)
from model_scripts.environment import RFPBackcastEnv, get_env
from model_scripts.agent_hierarchical_heuristic import DeterministicHA
import pandas as pd 

def main():
    rfp_case = "small"
    planning_horizon = 96
    allow_spot_buy = True
    solver = 'gurobi'
    
    # guideline = "hourly_target" # Promote a fixed production flow for all hours.
    guideline = "production_value" # Reward production of ammonia based on estimating internal value of ammonia.
    
    n_episodes = 1
    
    env_config = {"allow_spot_buy": allow_spot_buy, "balancing_market": False, "verbose": True, "load_data": True, "inflexible": True}
    env = get_env(RFPBackcastEnv, env_config=env_config, layout_file="rfp_layout - resized.xlsx")
    
    agent = DeterministicHA(env=env, solver=solver, planning_horizon=planning_horizon, guideline=guideline, documentation=False)
    
    forecast_model = "forecaster" # Options: ("forecaster", "prophet", "persistence")
    env_options = {"historical_data_path": "historical_data/clean_dataframes/backcasting_timeseries.csv",
                     "episode_start": pd.Timestamp('2017-01-01 00:00:00'),
                     "forecaster_type": forecast_model}

    experiment_name = "_".join(["backcasting", forecast_model, str(agent), guideline, "ph", str(planning_horizon), "spot", str(allow_spot_buy), rfp_case])
    print("Start experiment: ", experiment_name)
    stats, trajectories = train(env, agent, num_episodes=1, verbose=True, experiment_name=experiment_name, options=env_options)
    agent.close()
    print("Experiment done")


import cProfile
if __name__ == '__main__':
    cProfile.run("main()", "run_profiles/run_DeterministicSpotBuy_24.prof")
    # Example of how to read the profile results:
    import pstats
    prof = pstats.Stats("run_profiles/run_DeterministicSpotBuy_24.prof")
    prof.strip_dirs().sort_stats("cumtime").print_stats(10)
    # cProfile.py -- Profile Python programs

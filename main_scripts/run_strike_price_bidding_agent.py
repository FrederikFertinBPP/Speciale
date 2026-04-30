""" Necessary path addendum if we want to run this script not from the root. 
    To run from root call:
    python -m test_scripts.SCRIPTNAME """
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common_scripts import train
from model_scripts.environment import RFPRecourseEnv, get_env
from model_scripts.agent_hierarchical_heuristic import StrikePriceBiddingAgent

def main():
    rfp_case = "small"
    allow_spot_buy = True
    planning_horizon = 4*24
    n_strike_prices = 1
    n_scenarios = 1
    n_episodes = 1
    solver = 'gurobi'

    """ Set experiment name """
    guideline = "production_value" # Reward production of ammonia based on estimating internal value of ammonia.
    
    env_config = {"allow_spot_buy": allow_spot_buy, "balancing_market": True, "verbose": True, "load_data": True, "inflexible": True}
    env = get_env(RFPRecourseEnv, env_config=env_config, layout_file="rfp_layout - resized.xlsx")
    
    agent = StrikePriceBiddingAgent(env=env,
                            solver=solver,
                            documentation=False,
                            planning_horizon=planning_horizon,
                            guideline=guideline,
                            n_strike_prices=n_strike_prices,
                            n_sims=1,
                            n_scenarios=n_scenarios,
                            )
    

    experiment_name = "_".join(["test", str(agent), guideline, "ph", str(planning_horizon), "spot", str(allow_spot_buy), rfp_case])
    print("Start experiment: ", experiment_name)
    stats, trajectories = train(env, agent, experiment_name=experiment_name, num_episodes=n_episodes, verbose=True)
    # stats, trajectories = train(env, agent, num_episodes=1, verbose=True)
    agent.close()
    print("Experiment done")


import cProfile
if __name__ == '__main__':
    cProfile.run("main()", "run_profiles/run_StrikePriceAgent.prof")
    # Example of how to read the profile results:
    import pstats
    prof = pstats.Stats("run_profiles/run_StrikePriceAgent.prof")
    prof.strip_dirs().sort_stats("cumtime").print_stats(10)
    # cProfile.py -- Profile Python programs
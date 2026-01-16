""" Necessary path addendum if we want to run this script not from the root. 
    To run from root call in terminal: python -m test_scripts.SCRIPTNAME
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common_scripts import train # stats, trajectories = train(env, agent, num_episodes=n_episodes, verbose=True, seed=42)
from model_scripts.environment import RFPRecourseEnv, get_env
from model_scripts.agent_hierarchical_heuristic import RecourseAgent, StrikePriceBiddingAgent

def main():
    allow_spot_buy = False
    planning_horizon = 3 * 24
    solver = 'gurobi'
    n_scenarios = 1

    guideline = "production_value" # Reward production of ammonia based on estimating internal value of ammonia.
    isp_metric = 'mean'

    n_episodes = 20

    # scenarios = ["80percent_ammonia", "70percent_ammonia", ]
    # scenarios += ["60percent_ammonia", "50percent_ammonia", ]
    # scenarios += ["40percent_ammonia", "no_solar_power", "high_RE_level"]
    # scenarios = ["30percent_ammonia", "20percent_ammonia", ]
    scenarios = ["base_case_no_spot", "no_solar_power", "high_RE_level"]
    scenarios = ["no_solar_power", "high_RE_level"]
    scenarios = ["high_RE_level"]

    for scenario in scenarios:
        env = get_env(RFPRecourseEnv, allow_spot_buy=allow_spot_buy, balancing_market=True, verbose=True, load_data=True, scenario_name=scenario)
        
        agent = RecourseAgent(env=env, solver=solver, planning_horizon=planning_horizon, guideline=guideline,
                              n_scenarios=n_scenarios)
        # agent = RecourseAgent(env=env, solver=solver, planning_horizon=planning_horizon, guideline=guideline,
        #                     n_scenarios=n_scenarios, da_model_type="recourse DA")
        agent = StrikePriceBiddingAgent(env=env, solver=solver, planning_horizon=planning_horizon, guideline=guideline,
                            n_scenarios=n_scenarios, n_strike_prices=1, n_sims=1, isp_metric=isp_metric)
        
        experiment_name = "_".join(["scenario", scenario, str(agent), guideline, "ph", str(planning_horizon), "spot", str(allow_spot_buy)])#, "ispmetric", isp_metric])
        print("Start experiment: ", experiment_name)
        stats, trajectories = train(env, agent, experiment_name=experiment_name, num_episodes=n_episodes,
                                    verbose=True, continue_trajectories_and_stats=False,)
        agent.close()
        print("Experiment done")


import cProfile
if __name__ == '__main__':
    cProfile.run("main()", "run_profiles/run_DeterministicSpotBuy.prof")
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
    allow_spot_buy = True
    planning_horizon = 4*24
    n_strike_prices = 1
    n_scenarios = 1
    n_episodes = 50
    solver = 'gurobi'

    """ Set experiment name """
    guideline = "production_value" # Reward production of ammonia based on estimating internal value of ammonia.
    
    env = get_env(RFPRecourseEnv, allow_spot_buy=allow_spot_buy, balancing_market=True, verbose=True, load_data=True)
    
    agent = StrikePriceBiddingAgent(env=env,
                            solver=solver,
                            documentation=False,
                            planning_horizon=planning_horizon,
                            guideline=guideline,
                            n_strike_prices=n_strike_prices,
                            n_sims=5,
                            n_scenarios=n_scenarios,
                            )
    

    experiment_name = "_".join(["test", str(agent), guideline, "ph", str(planning_horizon), "spot", str(allow_spot_buy)])
    print("Start experiment: ", experiment_name)
    stats, trajectories = train(env, agent, experiment_name=experiment_name,
                                num_episodes=n_episodes, verbose=True, save_every=10)
    agent.close()
    print("Experiment done")


import cProfile
if __name__ == '__main__':
    cProfile.run("main()", "run_profiles/run_TrainDecisionRule.prof")
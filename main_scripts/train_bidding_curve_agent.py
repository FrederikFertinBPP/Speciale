""" Necessary path addendum if we want to run this script not from the root. 
    To run from root call:
    python -m test_scripts.SCRIPTNAME """
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common_scripts import train # stats, trajectories = train(env, agent, num_episodes=n_episodes, verbose=True, seed=42)
from model_scripts.environment import RFPYearEnv, get_env
from model_scripts.agent_hierarchical_heuristic import BiddingCurveAgent

def main():
    n_episodes = 10
    solver = 'gurobi'
    allow_spot_buy = True
    
    env_config = {"allow_spot_buy": allow_spot_buy, "balancing_market": False, "verbose": True, "load_data": True, "inflexible": False}
    env = get_env(RFPYearEnv, env_config=env_config, layout_file="rfp_layout - resized.xlsx")

    agent = BiddingCurveAgent(env=env, solver=solver, documentation=False, guideline=None,
                              n_price_domains=1)
    # agent = BiddingCurveAgent(env=env, solver=solver, documentation=False, guideline=None,
    #                           n_price_domains=2, domain_prices=[80],)
    # agent = BiddingCurveAgent(env=env, solver=solver, documentation=False, guideline=None,
    #                           n_price_domains=3, domain_prices=[80, 80*1.4],)
    
    experiment_name = "_".join(["train", str(agent), "spot", str(allow_spot_buy)])
    print("Start experiment: ", experiment_name)
    # stats, trajectories = train(env, agent, experiment_name=experiment_name,
    #                             num_episodes=n_episodes, verbose=True, save_agent=True)
    stats, trajectories = train(env, agent, num_episodes=10, verbose=True)
    agent.close()
    print("Experiment done")


import cProfile
if __name__ == '__main__':
    cProfile.run("main()", "run_profiles/train_DRAgent1.prof")
    # Example of how to read the profile results:
    import pstats
    prof = pstats.Stats("run_profiles/train_DRAgent1.prof")
    prof.strip_dirs().sort_stats("cumtime").print_stats(10)
    # cProfile.py -- Profile Python programs
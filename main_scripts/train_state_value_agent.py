""" Necessary path addendum if we want to run this script not from the root. 
    To run from root call:
    python -m test_scripts.SCRIPTNAME """
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from common_scripts import train # stats, trajectories = train(env, agent, num_episodes=n_episodes, verbose=True, seed=42)
from model_scripts.environment import RFPEnv, get_env
from model_scripts.agent_hierarchical_rl import StateValueHA

def main():
    n_episodes = 50
    seeds = [x for x in range(n_episodes)]
    solver = 'gurobi'

    initial_guess = []
    # guideline = "hourly_target" # Promote a fixed production flow for all hours.
    guideline = "production_value" # Reward production of ammonia based on estimating internal value of ammonia.
    initial_guess += [0, 0, 400]
    objective_logic = "value_maximization"
    initial_guess += [0, 0, 0, 0]

    ### Set experiment name and length ###
    experiment_name = "trainStateValueHA" + guideline + objective_logic
    
    env = get_env(RFPEnv, allow_spot_buy=True, balancing_market=False, verbose=True, load_data=True)

    epsilon = lambda steps, episodes: max(0.05, 1 - 2 * np.sqrt(steps) / 100)  # Epsilon decay function
    writer = SummaryWriter(f'runs/{experiment_name}')
    agent = StateValueHA(env=env,
                        writer=writer,
                        solver=solver,
                        documentation=False,
                        guideline=guideline,
                        objective_logic=objective_logic,
                        epsilon=epsilon,
                        gamma=0.9,
                        planning_horizon=24,
                        )
    agent.load(os.getcwd() + f"/models/rl_models/{experiment_name}")
    
    print("Start experiment: ", experiment_name)
    stats, trajectories = train(env, agent, experiment_name=experiment_name,
                                num_episodes=n_episodes, verbose=True, seed=seeds,
                                save_agent=True, continue_trajectories_and_stats=True)
    agent.close()
    print("Experiment done")


import cProfile
if __name__ == '__main__':
    cProfile.run("main()", "run_profiles/run_TrainDecisionRule.prof")
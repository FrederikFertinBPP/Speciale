""" Necessary path addendum if we want to run this script not from the root. 
    To run from root call:
    python -m test_scripts.SCRIPTNAME """
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common_scripts.RFP_initialization import create_rfp
from common_scripts import train # stats, trajectories = train(env, agent, num_episodes=n_episodes, verbose=True, seed=42)
from data_scripts.data_generator_v2 import DataForecaster
from model_scripts.environment import RFPShieldEnv
from model_scripts.agent_hierarchical_rl import DdpgHA
import numpy as np
from torch.utils.tensorboard import SummaryWriter

def main():
    n_episodes = 5
    planning_horizon = 4 * 24
    decision_horizon = 24
    seeds = [x for x in range(n_episodes)]

    forecaster = DataForecaster(from_pickle=True, cache_id="v2")
    forecaster = forecaster.unpickle()

    rfp = create_rfp()
    allow_spot_buy = True
    rfp.get_contract('Ammonia1').parameters['volume'] = rfp.get_component("Haber Bosch Plant").parameters.get('capacity') * 8760 / (2 if allow_spot_buy else 5)  # 50% capacity contracted
    
    initial_guess = []
    # guideline = "hourly_target" # Promote a fixed production flow for all hours.
    guideline = "production_value" # Reward production of ammonia based on estimating internal value of ammonia.
    initial_guess += [0, 0, 400]
    objective_logic = "value_maximization"
    initial_guess += [0, 0, 0, 0]
    
    env = RFPShieldEnv(rfp=rfp,
                       forecaster=forecaster,
                       decision_horizon=decision_horizon,
                       planning_horizon=planning_horizon,
                       allow_spot_buy=allow_spot_buy,
                       verbose=True,
                       guideline=guideline,
                       )

    """ Set experiment indicator """
    experiment_name = "testNewEnvRL" + guideline + objective_logic
    writer = SummaryWriter(f'runs/{experiment_name}')
    print("Start experiment: ", experiment_name)
    solver = 'gurobi'
    epsilon = lambda steps, episodes: max(0.05, 1 - 2 * np.sqrt(steps) / 100)  # Epsilon decay function
    agent = DdpgHA(env=env,
                   writer=writer,
                   solver=solver,
                   documentation=True,
                   guideline=guideline,
                   objective_logic=objective_logic,
                   epsilon=epsilon,
                   initial_guess=np.asarray(initial_guess),
                   )
    stats, trajectories = train(env, agent, experiment_name=experiment_name, num_episodes=n_episodes, verbose=True, seed=seeds, save_agent=True, load_agent=True)
    if len(solver.split('_')) > 1: # Probably gurobi_persistent
        agent.hourly_model.solver.close()
    print("Experiment done")


import cProfile
if __name__ == '__main__':
    cProfile.run("main()", "run_profiles/run_DeterministicSpotBuy.prof")
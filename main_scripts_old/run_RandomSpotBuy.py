""" Necessary path addendum if we want to run this script not from the root. 
    To run from root call:
    python -m test_scripts.SCRIPTNAME """
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common_scripts.RFP_initialization import create_rfp
from common_scripts import train # stats, trajectories = train(env, agent, num_episodes=n_episodes, verbose=True, seed=42)
from data_scripts.data_generator_v2 import DataForecaster
from model_scripts_old.RFP_operational_environment import SpotBuyRFPEnv
from model_scripts_old.rl_agents import RandomAgent

def main():
    n_episodes = 3
    decision_horizon = 24
    seeds = [x for x in range(n_episodes)]

    forecaster = DataForecaster(from_pickle=True, cache_id="v2")
    forecaster = forecaster.unpickle()

    rfp = create_rfp()
    rfp.get_contract('Ammonia1').parameters['volume'] = rfp.get_component("Haber-Bosch Plant").parameters.get('capacity') * 8760 / 2  # 20% capacity contracted
    env = SpotBuyRFPEnv(rfp=rfp, forecaster=forecaster, decision_horizon=decision_horizon)

    """ Set experiment indicator """
    experiment_name = "testRandomSpotBuy"
    print("Start experiment: ", experiment_name)
    agent = RandomAgent(env=env)
    stats, trajectories = train(env, agent, experiment_name=experiment_name, num_episodes=n_episodes, verbose=True, seed=seeds)
    print("Experiment done")


import cProfile
if __name__ == '__main__':
    cProfile.run("main()", "run_profiles/runRandomSpotBuy.prof")
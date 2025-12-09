""" Necessary path addendum if we want to run this script not from the root. 
    To run from root call:
    python -m test_scripts.SCRIPTNAME """
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common_scripts.RFP_initialization import create_rfp
from common_scripts import train # stats, trajectories = train(env, agent, num_episodes=n_episodes, verbose=True, seed=42)
from data_scripts.data_generator_v2 import DataForecaster
from model_scripts_old.RFP_operational_environment import RFPOperationalEnv
from model_scripts_old.rl_agents import StateAwareDDPGAgent, ContextAwareDDPGAgent, WarmStartDDPGAgent
from model_scripts_old.warmstarts import warmstart_agent
from gymnasium.wrappers import TimeAwareObservation, NormalizeObservation, FlattenObservation
import numpy as np

def main():
    n_episodes = 15
    planning_horizon = 1 * 24
    decision_horizon = 24
    seeds = [x for x in range(n_episodes)]

    rfp = create_rfp()
    rfp.get_contract('Ammonia1').parameters['volume'] = rfp.get_component("Haber-Bosch Plant").parameters.get('capacity') * 8760 / 5  # 20% capacity contracted

    forecaster = DataForecaster(from_pickle=True, cache_id="v2")
    forecaster = forecaster.unpickle()

    env = RFPOperationalEnv(rfp=rfp, forecaster=forecaster, decision_horizon=decision_horizon, planning_horizon=planning_horizon, normalize=True)
    s, info = env.reset()
    epsilon = lambda steps, episodes: max(0.05, 1 - 2 * np.sqrt(steps) / 100)  # Epsilon decay function
    warmstarting_agent = WarmStartDDPGAgent(env, alpha=0.003, epsilon=epsilon, batch_size=16, replay_buffer_size=10000, hidden_size=10, tau=0.1)
    warmstarting_agent = warmstart_agent(env, warmstarting_agent)
    agent = ContextAwareDDPGAgent(env, alpha=0.001, epsilon=epsilon, batch_size=16, replay_buffer_size=10000, hidden_size=10, warmstarting_agent=warmstarting_agent)
    
    """ Set experiment indicator """
    experiment_name = "testContextAwareRL"
    print("Start experiment: ", experiment_name)
    stats, trajectories = train(env, agent, experiment_name=experiment_name, num_episodes=n_episodes, verbose=True, seed=seeds)
    print("Experiment done")


import cProfile
if __name__ == '__main__':
    cProfile.run("main()", "run_profiles/run_RL.prof")
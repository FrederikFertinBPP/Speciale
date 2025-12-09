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
from model_scripts_old.agent_hierarchical_rl import DdpgHA
import numpy as np
from torch.utils.tensorboard import SummaryWriter

def main():
    n_episodes = 15
    planning_horizon = 4 * 24
    decision_horizon = 24
    seeds = [x for x in range(n_episodes)]

    rfp = create_rfp()
    rfp.get_contract('Ammonia1').parameters['volume'] = rfp.get_component("Haber-Bosch Plant").parameters.get('capacity') * 8760 / 5  # 20% capacity contracted

    forecaster = DataForecaster(from_pickle=True, cache_id="v2")
    forecaster = forecaster.unpickle()

    env = RFPOperationalEnv(rfp=rfp, forecaster=forecaster, decision_horizon=decision_horizon, planning_horizon=planning_horizon, normalize=True)
    s, info = env.reset()
    epsilon = lambda steps, episodes: max(0.05, 1 - 2 * np.sqrt(steps) / 100)  # Epsilon decay function

    experiment_name = "testStateAwareSteeringDDPG"
    writer = SummaryWriter(f'runs/{experiment_name}')
    agent = DdpgHA(env, writer, alpha=0.001, batch_size=32, hidden_size=10, epsilon=epsilon, gamma=0.995)

    """ Set experiment indicator """
    print("Start experiment: ", experiment_name)
    stats, trajectories = train(env, agent, experiment_name=experiment_name, num_episodes=n_episodes, verbose=True, seed=seeds, load_agent=True, save_agent=True)
    agent.writer.close()
    print("Experiment done")


import cProfile
if __name__ == '__main__':
    cProfile.run("main()", "run_profiles/run_steering.prof")
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
from model_scripts_old.agent_hierarchical_heuristic import StochasticHA

def main():
    n_episodes = 5
    planning_horizon = 4 * 24
    decision_horizon = 24
    seeds = [x for x in range(n_episodes)]

    forecaster = DataForecaster(from_pickle=True, cache_id="v2")
    forecaster = forecaster.unpickle()

    rfp = create_rfp()
    rfp.get_contract('Ammonia1').parameters['volume'] = rfp.get_component("Haber-Bosch Plant").parameters.get('capacity') * 8760 / 5 # 20% capacity contracted
    env = RFPOperationalEnv(rfp=rfp, forecaster=forecaster, decision_horizon=decision_horizon, planning_horizon=planning_horizon)

    """ Set experiment indicator """
    experiment = "testStochastic_"
    
    """ Set experiment type """
    # guideline = "planning_target"
    guideline = "strike_price"

    experiment_name = experiment + guideline
    print("Start experiment: ", guideline)
    agent = StochasticHA(env=env, guideline=guideline, n_scenarios=10, solver='gurobi_persistent')
    stats, trajectories = train(env, agent, experiment_name=experiment_name, num_episodes=n_episodes, verbose=True, seed=seeds)
    agent.hourly_model.solver.close()
    print("Experiment done")


import cProfile
if __name__ == '__main__':
    cProfile.run("main()", "run_profiles/run_Stochastic.prof")
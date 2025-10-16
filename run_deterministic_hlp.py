from common_scripts.RFP_initialization import create_rfp
from model_scripts import train
from data_scripts.data_generator_v2 import DataForecaster
from model_scripts.RFP_operational_environment import RFPOperationalEnv
from model_scripts.agent_hierarchical_heuristic import DeterministicHierarchicalAgent

def main():
    planning_horizon = 4 * 24
    decision_horizon = 24

    forecaster = DataForecaster(from_pickle=True, cache_id="v2")
    forecaster = forecaster.unpickle()

    rfp = create_rfp()
    env = RFPOperationalEnv(rfp=rfp, forecaster=forecaster, decision_horizon=decision_horizon, planning_horizon=planning_horizon, seed=42)

    guideline = "planning_target"
    agent_planning_target = DeterministicHierarchicalAgent(env=env, guideline=guideline)
    guideline = "strike_price"
    agent_strike_price = DeterministicHierarchicalAgent(env=env, guideline=guideline)

    n_episodes = 1
    stats, trajectories = train(env, agent_planning_target, experiment_name="test3_"+guideline, num_episodes=n_episodes, verbose=True, seed=42)
    print("Experiment 1 done")
    stats, trajectories = train(env, agent_strike_price, experiment_name="test3_"+guideline, num_episodes=n_episodes, verbose=True, seed=42)
    print("Experiment 2 done")
    # stats, trajectories = train(env, agent, num_episodes=n_episodes, verbose=True, seed=42)
    print("Tests done")


import cProfile
if __name__ == '__main__':
    cProfile.run("main()", "run_profiles/profile_output_planning_run.prof")
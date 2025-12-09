from common_scripts.utils import load_trajectories, load_stats
from model_scripts_old.rl_agents import BasicBuffer

def warmstart_agent(env, agent, iters=2e4, spotbuy=False):
    experiment_names = ("testStochastic_strike_price", "testDeterministic_strike_price") # Input experiments here with data that is suitable for training the neural network.
    if spotbuy:
        experiment_names = ("testStochasticSpotbuy_strike_price", "testDeterministicSpotbuy_strike_price")
    trajectories = []
    stats = []
    for name in experiment_names:
        trajectories += load_trajectories(name)
        stats += load_stats(name, csv_version=False)
    steps = 0
    for episode in range(len(trajectories)):
        print("Warmstart episode:", episode+1, ". Out of:", len(trajectories))
        stat = stats[episode]
        trajectory = trajectories[episode]
        for ix in range(len(trajectory.reward)):
            reward = trajectory.reward[ix]
            actions = trajectory.action[ix]
            normalized_actions = env.action_scaler.transform(actions)
            state = trajectory.state[ix]
            normalized_state = env.state_scaler.transform([state])[0]
            info = trajectory.env_info[ix]
            next_state = trajectory.state[ix+1]
            normalized_next_state = env.state_scaler.transform([next_state])[0]
            next_info = trajectory.env_info[ix+1]
            done = bool(ix == len(trajectory.reward)-1)
            agent.train(s=normalized_state, a=normalized_actions, r=float(reward), sp=normalized_next_state, done=done, info_s=info, info_sp=next_info)
            steps += 1
    while steps < iters: # Fit heavily on trajectories.
        agent.experience_replay()
        steps += 1
    agent.warmstart = False
    print("Warmstart done")
    return agent
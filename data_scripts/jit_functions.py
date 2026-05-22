from numba import jit
import numpy as np

@jit(nopython=True)
def weighted_choice(p):
    cumsum = np.cumsum(p)
    r = np.random.random()
    for i in range(len(cumsum)):
        if r < cumsum[i]:
            return i
    return len(p) - 1  # fallback

@jit(nopython=True)
def simulate_stochastic_wind_process(horizon, ma_lag, observations, differences, domains,
                                 p5_obs, p98_obs, pos_dist, neg_dist,
                                 sigma_laplace, pol_model_pos, pol_model_neg, pol_model_mode):
    sim_cf = np.zeros(horizon + ma_lag)
    deltas = np.zeros(horizon + ma_lag)
    directions = np.zeros(horizon + ma_lag)

    sim_cf[:ma_lag] = observations
    deltas[:ma_lag] = differences
    directions[:ma_lag] = (differences > 0).astype(np.int32) - (differences <= 0).astype(np.int32)

    previous_diff_direction = directions[ma_lag - 1]

    for t in range(ma_lag, horizon + ma_lag):
        if (directions[t] == 0 or
            (sim_cf[t - 1] <= p5_obs and directions[t] == -1) or
            (sim_cf[t - 1] >= p98_obs and directions[t] == 1)):

            current_domain = max(0, np.sum(domains <= sim_cf[t - 1]) - 1) # np.argwhere(domains <= sim_cf[t - 1])[-1][0]
            p = neg_dist[:, current_domain] if previous_diff_direction == 1 else pos_dist[:, current_domain]
            interval_length = weighted_choice(p) # np.random.choice(np.arange(len(p)), p=p)
            directions[t:min(t + interval_length, horizon)] = -previous_diff_direction
            previous_diff_direction *= -1

        ma = np.mean(sim_cf[t - ma_lag:t])
        direction = directions[t - ma_lag]
        mean = pol_model_pos @ np.asarray([1,ma,ma**2]) if direction == 1 else pol_model_neg  @ np.asarray([1,ma,ma**2])
        # mode = pol_model_mode @ np.asarray([1,ma]) # unused
        delta = direction * np.random.exponential(max(sigma_laplace / 2, mean))
        delta = max(delta, 0) if direction == 1 else min(delta, 0)

        deltas[t] = delta
        sim_cf[t] = sim_cf[t - 1] + delta

    return sim_cf, deltas

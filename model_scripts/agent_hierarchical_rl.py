import numpy as np
import pandas as pd
import gymnasium as gym

from common_scripts.utils import Flattener
from model_scripts.hourly_models import HourlyDeterministicLPModel
from model_scripts.environment import RFPShieldEnv, RFPEnv
from model_scripts.rl_agents import NN_DDPGAgent
from model_scripts.agent_hierarchical_heuristic import HierarchicalAgent, DeterministicHA


class SteeringEnv(gym.Env):
    """ Used env attributes (By DDPG agents):
    action_space.shape,
    observation_space,
    decision_horizon,
    forecaster,
    """
    def __init__(self, env, steering_actions):
        self.observation_space = gym.spaces.Dict({
                        'time': gym.spaces.Box(low=0, high = np.asarray([1,1]), dtype = np.float64), # Relative day of month, relative day of year
                        'nh3soc': gym.spaces.Box(low=0, high = 1, dtype = np.float64), # Time of month X SOC
                        'nh3status': gym.spaces.Box(low=0, high = 1, dtype = np.float64), # Time of year X Contract status
                        'ship_available': gym.spaces.MultiBinary(n=1), # Whether there is an ammonia ship available within the planning_horizon
                        })
        # self.observation_space = env.observation_space
        self.action_names = list(steering_actions.keys())
        self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.action_names),), dtype = np.float64)


class DdpgHA(HierarchicalAgent):
    guideline_options = ('production_value', None)

    def __init__(self,
                 env:RFPShieldEnv,
                 writer=None,
                 guideline = "production_value",
                 hourly_model_class=HourlyDeterministicLPModel,
                 steering_agent_class=NN_DDPGAgent,
                 solver = 'gurobi',
                 documentation:bool = False,
                 objective_logic:str|None = "value_maximization",
                 epsilon=None,
                 **kwargs,
                 ):
        super().__init__(env=env,
                         writer=writer,
                         guideline=guideline,
                         hourly_model_class=hourly_model_class,
                         solver=solver,
                         documentation=documentation,
                         objective_logic=objective_logic,
                         **kwargs,
                         )
        self.steering_action_space = self.hourly_model.steering_variables
        self.steering_actions_flat = Flattener().flatten(self.steering_action_space)
        self.steering_action = None

        self.steering_env = SteeringEnv(env, self.steering_actions_flat)
        self.steering_agent = steering_agent_class(self.steering_env,
                                                   writer = self.writer,
                                                   epsilon = epsilon,
                                                   **kwargs)

        self.logbook = {name: [] for name in self.steering_actions_flat.keys()}

    def _update_logbook(self):
        for key in self.steering_actions_flat.keys():
            self.logbook[key].append(self.steering_actions_flat[key])

    def _set_steering_action(self, obs, k, info:dict):
        action = self.steering_agent.pi(obs, k, info_s=info)
        for ix, key in enumerate(self.steering_actions_flat.keys()):
            self.steering_actions_flat[key] = action[ix]
        self.steering_action = Flattener().unflatten(self.steering_actions_flat)

    def _solve_hourly_decisions(self, obs, t:pd.Timestamp, info:dict):
        forecasts, electricity_price_forecast = self._get_forecasts_and_electricity(t)
        supplier_cf = self._get_supplier_cf(obs, forecasts[0])
        datetime_data = {t: forecasts[0].index[t] for t in range(self.env.planning_horizon)}

        offtaker_availability = {(self.env.offtaker_names[ix], t) : obs['context']['offtakers'][t,ix]
                                 for ix in range(len(self.env.offtaker_names)) for t in range(self.planning_horizon)}

        data = { # Set up the necessary data for the LP Concrete Model
            None: {
                'T_datetime': datetime_data, 
                'init_soc': dict(zip(self.env.storage_names, obs['state']['storages'])),
                'supplier_cf': supplier_cf,
                'init_contract_status' : dict(zip(self.env.contract_names, obs['state']['contracts'])),
                'electricity_price': electricity_price_forecast,
                'offtaker_availability': offtaker_availability,
                **self.steering_action,
            }
        }

        # Solve hourly LP model
        self.hourly_model.build_concrete_instance(data=data)
        self.hourly_model.run(verbose=False)

        actions = self.hourly_model.get_actions()
        if actions is not None:
            return actions
        else:
            self._save_obs_for_debug(obs, info)
            raise ValueError(f"Could not get actions.\nTime: {t}.\nState at termination: {obs['state']}.")

    def pi(self, obs, k, info:dict):
        """ Hierarchical policy for the agent. We start by defining the guidelines for the hourly decisions. """
        t = info["time"]

        self._set_steering_action(obs, k=0, info=info)

        hourly_actions = self._solve_hourly_decisions(obs, t, info=info) # Day-ahead solving

        self._update_logbook()

        return np.asarray(hourly_actions)

    def train(self, s, a, r, sp, done=False, info_s=None, info_sp=None):
        self.steering_agent.train(s, self.steering_action, r, sp, done, info_s, info_sp)

    def save(self, path):
        self.steering_agent.save(path)

    def load(self, path):
        self.steering_agent.load(path)


class StateValueHA(DeterministicHA):
    guideline_options = ('production_value', None)

    def __init__(self,
                 env:RFPEnv,
                 writer=None,
                 guideline = "production_value",
                 hourly_model_class=HourlyDeterministicLPModel,
                 steering_agent_class=NN_DDPGAgent,
                 solver = 'gurobi',
                 documentation:bool = False,
                 objective_logic:str|None = "value_maximization",
                 epsilon=None,
                 **kwargs,
                 ):
        super().__init__(env=env,
                         writer=writer,
                         guideline=guideline,
                         hourly_model_class=hourly_model_class,
                         solver=solver,
                         documentation=documentation,
                         objective_logic=objective_logic,
                         **kwargs,
                         )
        self.steering_action_space = self.hourly_model.steering_variables[objective_logic]
        self.steering_actions_flat = Flattener().flatten(self.steering_action_space)
        self.steering_action = None

        self.steering_env = SteeringEnv(env, self.steering_actions_flat)
        self.steering_agent = steering_agent_class(self.steering_env,
                                                   writer = self.writer,
                                                   epsilon = epsilon,
                                                   **kwargs)

        for name in self.steering_actions_flat.keys():
            self.logbook[name] = []

    def _update_logbook(self):
        for key in self.steering_actions_flat.keys():
            self.logbook[key].append(self.steering_actions_flat[key])

    def _get_steering_obs(self, obs):
        time_context = obs['context']['time'][0][2:4]
        nh3_ix = np.where(np.asarray(self.env.storage_names) == 'Ammonia Storage')[0][0]
        storage_soc = obs['state']['storages'][nh3_ix] / self.env.storage_state_space.high[nh3_ix]
        nh3_ix = np.where(np.asarray(self.env.contract_names) == 'Ammonia1')[0][0]
        status = obs['state']['contracts'][nh3_ix] / self.env.contract_state_space.high[nh3_ix]
        ship_available = int(bool(sum(obs['context']['offtakers'][:self.planning_horizon,1])))
        return {'time': time_context, 'nh3soc': storage_soc, 'nh3status': status, 'ship_available': ship_available,}

    def _set_steering_action(self, obs, k, info:dict):
        s_obs = self._get_steering_obs(obs)
        action = self.steering_agent.pi(s_obs, k, info_s=info)
        for ix, key in enumerate(self.steering_actions_flat.keys()):
            self.steering_actions_flat[key] = action[ix]
        self.steering_action = Flattener().unflatten(self.steering_actions_flat)

    def _solve_hourly_decisions(self, obs, time:pd.Timestamp, info:dict):
        data = self._construct_concrete_data(obs, time)
        forecasts, electricity_price_forecast = self._get_forecasts_and_electricity(time)
        supplier_cf = self._get_supplier_cf(obs, forecasts[0])

        data[None]["supplier_cf"] = supplier_cf
        data[None]["electricity_price"] = electricity_price_forecast
        data[None] = {**data[None], **self.steering_action}

        # Solve hourly LP model
        self.hourly_model.build_concrete_instance(data=data)
        return self._run_hourly_model(obs, time, info)

    def pi(self, obs, k, info:dict):
        """ Hierarchical policy for the agent. We start by defining the guidelines for the hourly decisions. """
        time = info["time"]
        if self.guideline == 'hourly_target':
            # Constant heuristic:
            self._calculate_hourly_ammonia_target(obs, time)
        else:
            if time.day % 15 == 1 and self.guideline == 'production_value': # We do not expect big changes in strike price throughout the year - update two times a month.
                self.ammonia_strike_price = self._estimate_strike_price(obs=obs, time=time, info=info, n_sims=self.n_sims, metric='mean')

        self._set_steering_action(obs, k=0, info=info)

        hourly_actions = self._solve_hourly_decisions(obs, time, info=info) # Day-ahead solving

        self._update_logbook()

        return np.asarray(hourly_actions)

    def train(self, s, a, r, sp, done=False, info_s=None, info_sp=None):
        steering_obs = self._get_steering_obs(s)
        steering_obs_p = self._get_steering_obs(sp)
        self.steering_agent.train(steering_obs, self.steering_action, r, steering_obs_p, done, info_s, info_sp)

    def save(self, path):
        self.steering_agent.save(path)

    def load(self, path):
        self.steering_agent.load(path)



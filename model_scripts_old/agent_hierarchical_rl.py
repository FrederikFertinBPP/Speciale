import numpy as np
import pandas as pd
import gymnasium as gym

from model_scripts_old.lp_deterministic import HourlyDeterministicLPModel, SpotBuyHourlyDeterministicLPModel
from model_scripts_old.lp_stochastic import HourlyStochasticLPModel, SpotBuyHourlyStochasticLPModel
from model_scripts_old.RFP_operational_environment import RFPOperationalEnv, SpotBuyRFPEnv

from model_scripts_old.RFP_operational_environment import RFPOperationalEnv
from model_scripts_old.rl_agents import SteeringDDPGAgent
from model_scripts_old.agent_hierarchical_heuristic import HierarchicalAgent

class SteeringEnv(gym.Env):
    """ Used env attributes (By DDPG agents):
    action_space.shape,
    observation_space,
    decision_horizon,
    forecaster,
    """
    def __init__(self, env):
        self.env_ = env
        self.action_names = [name for name, contract in self.env_.rfp.get_contracts() if contract.target_frequency != 'hourly']
        self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.action_names),), dtype = np.float64)
        self.state_space = self.env_.state_space
        self.decision_horizon = self.env_.decision_horizon # Used by context aware DDPG agent.
        self.forecaster = self.env_.forecaster # Point to the same forecaster. This assures that the steering env and the parent env are synchronized.


class LinearAgent: ...


class LinearDRHA(HierarchicalAgent):
    def __init__(self,
                 env:RFPOperationalEnv,
                 guideline = "strike_price",
                 hourly_model_class=HourlyDeterministicLPModel,
                 steering_agent_class=LinearAgent,
                 solver='gurobi',
                 documentation=False,
                 **kwargs,
                 ):
        super().__init__(env=env, guideline=guideline, hourly_model_class=hourly_model_class, solver=solver, documentation=documentation, **kwargs)

        self.electricity_consumption = {}
        self.electricity_consumption['hydrogen'] = self.env.rfp.get_component('Electrolyzer').parameters.get('rate', 1/50) # tH2/MWh
        self.electricity_consumption['ammonia'] = self.electricity_consumption['hydrogen'] * self.env.rfp.get_component('Haber Bosch Plant').parameters.get('rate', 5.5) # tNH3/MWh
        self.electricity_consumption['ammonia'] = 1/self.electricity_consumption['ammonia'] + self.env.rfp.get_component('Haber Bosch Plant').parameters.get('electricity_consumption', 1) # MWh/tNH3
        self.electricity_consumption['hydrogen'] = 1/self.electricity_consumption['hydrogen'] # MWh/tH2

        self.steering_env = SteeringEnv(self.env)
        epsilon = lambda steps, episodes: max(0.05, 1 - 2 * np.sqrt(steps) / 100) # Epsilon decay function
        self.steering_agent = steering_agent_class(self.steering_env, gamma=0.995, epsilon=epsilon, hidden_size=10, batch_size=64)

        self.hydrogen_hourly_target = self.env.rfp.get_contract('Hydrogen1').parameters.get('volume', 1) # tH2/h

        self.production_strike_prices = {name: None for name, contract in self.env.rfp.get_contracts() if contract.target_frequency != 'hourly'}
        
        self.logbook = {name: [] for name in self.production_strike_prices.keys()}


class DdpgHA(HierarchicalAgent):
    guideline_options = ('production_value', None)
    
    def __init__(self,
                 env:RFPOperationalEnv,
                 writer=None,
                 guideline = "production_strike_prices",
                 hourly_model_class=HourlyDeterministicLPModel,
                 steering_agent_class=SteeringDDPGAgent,
                 solver='gurobi',
                 documentation=False,
                 **kwargs,
                 ):
        super().__init__(env=env, writer=writer, guideline=guideline, hourly_model_class=hourly_model_class, solver=solver, documentation=documentation, **kwargs)

        self.production_strike_prices = {name: contract.parameters.get('price', None) for name, contract in self.env.rfp.get_contracts() if contract.target_frequency != 'hourly'}

        self.steering_env = SteeringEnv(self.env)
        self.steering_agent = steering_agent_class(self.steering_env,
                                                   initial_guess = np.asarray(list(self.production_strike_prices.values())),
                                                   writer = self.writer,
                                                   **kwargs)

        self.hydrogen_hourly_target = self.env.rfp.get_contract('Hydrogen1').parameters.get('volume', 1) # tH2/h

        self.steering_action = np.asarray(list(self.production_strike_prices.values()), dtype=float)

        self.logbook = {name: [] for name in self.production_strike_prices.keys()}

    def _update_logbook(self):
        for key in self.production_strike_prices.keys():
            self.logbook[key].append(self.production_strike_prices[key])

    def _set_production_strike_prices(self, s, k, info:dict):
        steering_action = self.steering_agent.pi(s, k, info_s=info)
        for ix, key in enumerate(self.production_strike_prices.keys()):
            self.production_strike_prices[key] = steering_action[ix]
        return steering_action

    def _solve_hourly_decisions(self, s, t:pd.Timestamp, info:dict):
        # Forecast prices and renewables for the planning horizon
        forecasts = self.env.forecaster.forecast(start=t, end=t+pd.Timedelta(self.env.planning_horizon-1, 'h'), n_forecasts=1) # list of DFs
        wind_profile = self.env.wind_mapper(forecasts[0]['wind'])
        wind_profile.loc[info['asset_wind_realization'].index] = info['asset_wind_realization']['wind']
        solar_profile = self.env.solar_mapper(forecasts[0]['solar'])
        solar_profile.loc[info['asset_solar_realization'].index] = info['asset_solar_realization']['solar']

        # Reformat for PYOMO model:
        wind_cf = {('WindPower', t): wind_profile.iloc[t] for t in range(self.env.planning_horizon)}
        solar_cf = {('SolarPower', t): solar_profile.iloc[t] for t in range(self.env.planning_horizon)}
        nuclear_cf = {('NuclearPower', t): 1.0 for t in range(self.env.planning_horizon)}
        electricity_price = {t: forecasts[0].iloc[t]['price'] for t in range(self.env.planning_horizon)}

        # Set up the necessary data for the LP Concrete Model

        data = {
            None: {
                'T_datetime': {t: solar_profile.index[t] for t in range(self.env.planning_horizon)}, 
                'init_soc': {
                    'Hydrogen Storage': info['final_soc_H2'],
                    'Ammonia Storage': info['final_soc_NH3'],
                },
                'supplier_cf': {
                    **wind_cf,
                    **solar_cf,
                    **nuclear_cf,
                },
                'electricity_price': electricity_price,
            }
        }

        # Solve hourly LP:
        self.hourly_model.build_concrete_instance(data=data)
        self.hourly_model.run(verbose=False)
        return self.hourly_model.get_actions()
    
    def pi(self, s, k, info:dict):
        """ Hierarchical policy for the agent. We start by defining the guidelines for the hourly decisions. """
        t = info["time"]

        self.steering_action = self._set_production_strike_prices(s=s, k=0, info=info)

        hourly_actions = self._solve_hourly_decisions(s=s, t=t, info=info) # Day-ahead solving

        self._update_logbook()

        return np.asarray(hourly_actions)

    def train(self, s, a, r, sp, done=False, info_s=None, info_sp=None):
        self.steering_agent.train(s, self.steering_action, r, sp, done, info_s, info_sp)

    def save(self, path):
        self.steering_agent.save(path)
    
    def load(self, path):
        self.steering_agent.load(path)

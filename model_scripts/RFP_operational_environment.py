import gymnasium as gym
import numpy as np
import pandas as pd
from common_scripts.utils import cache_read
from common_scripts.RFP_initialization import RenewableFuelPlant
from data_scripts.data_generator_v2 import DataForecaster
from sklearn.preprocessing import MinMaxScaler
import copy
from dateutil.relativedelta import relativedelta
import os


class VRESystemToAssetMapping():
    def __init__(self, model):
        self.model = model
    
    def __call__(self, *args, **kwds):
        return np.clip(self.model(*args, **kwds), 0, 1)


class RFPOperationalEnv(gym.Env):
    error_margin = 1e-5

    def __init__(self,
                 rfp:RenewableFuelPlant,
                 forecaster:DataForecaster = None,
                 decision_horizon:int = 24,
                 planning_horizon:int = 4*24, 
                 dt:int = 1,
                 normalize:bool = False,
                 verbose:bool = False,
                 seed:int = None,
                 ):
        """
        Initialize the RFP environment.
        This environment simulates a hypothetical path planning problem.
        """
        self.rfp = rfp
        self.original_forecaster = forecaster
        self.planning_horizon = planning_horizon # Planning horizon in hours
        self.decision_horizon = decision_horizon # Decision horizon in hours
        # Whether to normalize the state and action spaces (dependent on the algorithm used for decision-making)
        # Should maybe be set by the agent instead?
        # But also important to have defined in the environment when assessing violation penalties.
        self.normalize_step = normalize 
        self.dt = dt  # Time step in hours
        self.verbose = verbose
        self.seed = seed # Random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)

        self.state = None # Placeholder for the state of the environment
        self.normalized_state = None # Placeholder for the normalized state of the environment
        self.time = None # Placeholder for the current time in the environment
        self.context = None # Placeholder for the context of the environment (stochastic forecasts)
        self.technical_violation_cost = 1 # Penalty for technical violations in the decision making
        
        """ Non-changing Plant Specifications """
        self.kappa_hb           = self.rfp.get_component("Haber-Bosch Plant").parameters.get("rate", 0)         # Conversion rate of Haber-Bosch [tNH3/tH2]
        self.eta_electrolyzer   = self.rfp.get_component("Electrolyzer").parameters.get("rate", 1/50)           # Efficiency of the electrolyzer [tH2/MWh]
        self.eta_hb             = self.rfp.get_component("Haber-Bosch Plant").parameters.get("charge_rate", 0)  # Electricity usage of the Haber-Bosch plant [MWh/tNH3]
        self.eta_h2_storage     = self.rfp.get_component("Hydrogen Storage").parameters.get("charge_rate", 0)   # Power consumed by the compressor to fill the H2 storage.
        self.ship_capacity      = self.rfp.get_component("Ammonia Shipment").parameters.get("capacity", 20000)  # tNH3 of capacity on ship
        self.wind_capacity      = self.rfp.get_component("WindPower").parameters.get('capacity', 200)           # Wind capacity in MW
        self.solar_capacity     = self.rfp.get_component("SolarPower").parameters.get('capacity', 400)          # Solar capacity in MWp
        self.gcp_capacity       = self.rfp.get_component("GCP").parameters.get('capacity', 700)                 # Grid-connection capacity in MW

        """ Define the action space """
        # The action space consists of four actions:
        # 1. Amount of electricity to sell to the grid (GCP)
        # 2. Amount of hydrogen to sell through the H2 pipeline
        # 3. Amount of ammonia to store in the ammonia storage
        # The action space is defined as a Box with lower and upper bounds based on the capacities of the components
        self.action_names = ['Grid Power Sale [MW]', 'Electrolyzer Power [MW]', "H2 Pipeline Flow [tH2/h]", "NH3 Production [tNH3/h]", "NH3 Spot Sale [tNH3]"]
        self.action_space = gym.spaces.Box( low  = np.asarray([[0, 0, 0, 0, 0]] * self.decision_horizon, dtype = np.float64),
                                            high = np.asarray([[self.gcp_capacity,
                                                                self.rfp.get_component("Electrolyzer").parameters['capacity']/self.eta_electrolyzer,
                                                                self.rfp.get_component("Hydrogen Pipeline").parameters['capacity'],
                                                                self.rfp.get_component("Haber-Bosch Plant").parameters['capacity'],
                                                                self.rfp.get_contract("AmmoniaSpot").parameters['cap']]] * self.decision_horizon,
                                                                dtype = np.float64),
                                            shape = (self.decision_horizon, len(self.action_names)),
                                            dtype = np.float64)
        self.action_space.low_ = self.action_space.low[0]
        self.action_space.high_ = self.action_space.high[0]
        
        """ Define the state space """
        self.state_names, s_low, s_high = [], [], []
        for name, comp in self.rfp.get_components():
            if comp.is_storage:
                self.state_names.append(name)
                s_high.append(comp.parameters["capacity"])
                s_low.append(0)
        for name, cont in self.rfp.get_contracts():
            if cont.target_frequency != "hourly": # Non-hourly contracts are a part of the observation space as this info is needed for the agent to make decisions
                if cont.target_frequency is not None:
                    s_low.append(0)
                    self.state_names.append(name)
                    s_high.append(cont.parameters.get('volume', 1e10)) # Set large upper bound for non-hourly contracts
                elif cont.parameters.get("shipment_frequency", None) is not None:
                    s_low.append(0)
                    self.state_names.append(name)
                    s_high.append(cont.parameters.get('cap', 1e10))
        # The state space is defined as a Box with lower and upper bounds based on the capacities
        self.state_space = gym.spaces.Box(low   = np.asarray(s_low),
                                          high  = np.asarray(s_high),
                                          dtype = np.float64)
        
        """ Define the context of the state space, which are the stochastic forecasts. """
        self.context_names = self.rfp.uncertainties
        el_price_bounds = (-500, 5000) # €/MWh price limits: https://www.acer.europa.eu/monitoring/electricity_market_integration_2024
        self.context_space = gym.spaces.Box(low   = np.asarray([[0, 0, el_price_bounds[0]]] * self.planning_horizon, dtype=np.float64),
                                            high  = np.asarray([[self.wind_capacity, self.solar_capacity, el_price_bounds[1]]] * self.planning_horizon, dtype=np.float64),
                                            shape = (self.planning_horizon, len(self.context_names)),
                                            dtype = np.float64)
        self.context_space.low_ = self.context_space.low[0]
        self.context_space.high_ = self.context_space.high[0]
        
        """ We define the observation space as the union of the space and context. """
        self.observation_space = gym.spaces.Dict({
            "state": self.state_space,
            "context": self.context_space
        })
        # Define scalers for normalizing the state, context, and action spaces
        self.state_scaler = MinMaxScaler(feature_range=(0, 1))
        self.state_scaler.fit(np.vstack([self.state_space.low, self.state_space.high]))
        self.action_scaler = MinMaxScaler(feature_range=(0, 1))
        self.action_scaler.fit(np.vstack([self.action_space.low, self.action_space.high]))
        self.context_scaler = MinMaxScaler(feature_range=(0, 1))
        self.context_scaler.fit(np.vstack([self.context_space.low, self.context_space.high]))

        """ Retrieve mappers, which produce VRE profiles for single assets, given a system production profile. """
        cache_path_mappers = os.getcwd() + "/models/plant_models/"
        solar_mapper = cache_read(cache_path_mappers + "solar.pkl")
        self.solar_mapper = VRESystemToAssetMapping(solar_mapper)
        wind_mapper = cache_read(cache_path_mappers + "wind.pkl")
        self.wind_mapper = VRESystemToAssetMapping(wind_mapper)

    def step(self, action):
        if self.normalize_step:
            return self._normalized_step(action)
        else:
            return self._step(action)

    def _normalized_step(self, normalized_action):
        """
        Perform a step in the environment with the given normalized action.
        Args:
            normalized_action (np.array): The normalized action in the range [0, 1].
        Returns:
            tuple: A tuple containing the next state, reward, terminated flag, truncated flag, and additional info.
        """
        # Denormalize the action to the original action space
        action = self.action_scaler.inverse_transform(normalized_action)
        state, reward, terminated, truncated, info = self._step(action)
        normalized_state = self.state_scaler.transform(state)
        return normalized_state, reward, terminated, truncated, info

    def _step(self, action):
        """
        Perform a step in the environment with the given action.
        Args:
            action np.array: The actions to take. shape:(decision_horizon, n_actions)
        Returns:
            tuple: A tuple containing the next state, reward, terminated flag, truncated flag, and additional info.
        """
        truncated = False
        terminated = False
        technical_violations = []
        def _clip(a, a_min, a_max):
            clipped = np.clip(a, a_min=a_min, a_max=a_max)
            violations = (clipped - a) / (a_max - a_min)  # Calculate the normalized penalty for deviating from the action space bounds
            if type(violations) is not np.ndarray:
                technical_violations.extend(np.asarray([violations]))
            else:
                technical_violations.extend(violations)
            return clipped, violations
        
        """ Get action, state, and stochastic realizations. """
        # Ensure the action is within the action space bounds
        clipped_action, action_technical_violations = _clip(action, self.action_space.low, self.action_space.high)
        
        # Current state of the environment
        soc_h2, soc_nh3, ytd_nh3, nh3_spot_sold_this_month = self.state
        # Actions taken by the agent
        # (1) Amount of electricity to sell to the grid (2) Power flowing to the electrolyzer
        # (3) Amount of hydrogen to sell through the H2 pipeline (4) Amount of ammonia to store
        p_to_grid, p_electrolyzer, m_h2_to_pipeline, m_nh3_to_storage, m_nh3_spot_sale = np.transpose(clipped_action)
        p_balancing = np.zeros(self.decision_horizon)
        soc_h2_list = np.zeros(self.decision_horizon)
        soc_nh3_list = np.zeros(self.decision_horizon)
        m_nh3_to_ship = 0
        
        # Realize the stochasticities (realizing prices, loading VRE which is realized in the previous step):
        p_wind, p_solar, p_base, prices = self.get_power_and_price_step(self.time)
        procured_power = p_wind + p_solar + p_base
        
        # Spot sale of ammonia
        m_nh3_spot_sale = np.sum(m_nh3_spot_sale)
        nh3_spot_sold_this_month, nh3_spot_sale_adjustment = _clip(m_nh3_spot_sale + nh3_spot_sold_this_month, self.state_space.low[3], self.state_space.high[3])
        m_nh3_spot_sale += nh3_spot_sale_adjustment # If our spot market offtaker does not want to buy more this month - cap it.

        """ The environment step is a number of hours where we have a predefined collection of decisions/actions. Typically 24 hours. """
        for t in range(self.decision_horizon):
            soc_nh3, nh3_flow_adjustment = _clip(m_nh3_to_storage[t] + soc_nh3, self.state_space.low[1], self.state_space.high[1])  # Update the state of charge of ammonia
            m_nh3_to_storage[t] += nh3_flow_adjustment  # Adjust the flow of ammonia to storage based on the action taken
            
            m_h2_to_hb = m_nh3_to_storage[t] / self.kappa_hb  # Amount of hydrogen to send to the Haber-Bosch plant

            m_h2_from_electrolyzer = p_electrolyzer[t] * self.eta_electrolyzer  # Amount of hydrogen produced by the electrolyzer
            m_h2_to_storage = m_h2_from_electrolyzer - m_h2_to_pipeline[t] - m_h2_to_hb # Amount of hydrogen to store in the H2 storage
            soc_h2, h2_flow_adjustment = _clip(m_h2_to_storage + soc_h2, self.state_space.low[0], self.state_space.high[0])  # Update the state of charge of hydrogen
            m_h2_to_storage += h2_flow_adjustment  # Adjust the flow of hydrogen to storage if needed
            # If the hydrogen storage could not handle the flows determined, we first adjust the electrolyzer, then the pipeline, and finally the Haber-Bosch plant
            if np.abs(h2_flow_adjustment) > self.error_margin:
                p_electrolyzer[t], pipeline_adjustment = _clip(p_electrolyzer[t] + h2_flow_adjustment / self.eta_electrolyzer, self.action_space.low_[1], self.action_space.high_[1])
                if np.abs(pipeline_adjustment) > self.error_margin:
                    m_h2_to_pipeline[t], hb_adjustment = _clip(m_h2_to_pipeline[t] + pipeline_adjustment * self.eta_electrolyzer, self.action_space.low_[2], self.action_space.high_[2])
                    if np.abs(hb_adjustment) > self.error_margin:
                        m_nh3_to_storage[t], warn1 = _clip(m_nh3_to_storage[t] - hb_adjustment / self.kappa_hb, self.action_space.low_[3], self.action_space.high_[3])
                        soc_nh3, warn2 = _clip(soc_nh3 - hb_adjustment / self.kappa_hb, self.state_space.low[1], self.state_space.high[1])
                        truncated = np.abs(warn1) > self.error_margin or np.abs(warn2) > self.error_margin

            p_to_hb = m_nh3_to_storage[t] * self.eta_hb  # Amount of electricity to send to the Haber-Bosch plant
            p_compressor = (m_h2_to_storage > 0) * m_h2_to_storage * self.eta_h2_storage # No power used if the storage is discharged.

            # Potential power imbalance, settled in intraday market at unfavourable prices.
            p_balancing[t] = procured_power[t] - p_to_grid[t] - p_electrolyzer[t] - p_to_hb - p_compressor
            
            if self.time.is_month_end and t==self.decision_horizon-1: # The last day of the month a shipment comes.
                m_nh3_to_ship += min(self.ship_capacity, soc_nh3) # We ship everything we have, which fits on the ship.
                soc_nh3 -= m_nh3_to_ship # Update the state of charge of ammonia
                nh3_spot_sold_this_month = 0 # Spot market deal refreshes.
            soc_h2_list[t] = soc_h2
            soc_nh3_list[t] = soc_nh3
         
        ytd_nh3 += np.sum(m_nh3_to_storage) - m_nh3_spot_sale # Update the year-to-date ammonia produced for annual contract

        """ Log the state of the final hour """
        self.state = np.asarray([soc_h2, soc_nh3, ytd_nh3, nh3_spot_sold_this_month], dtype=np.float64)

        """ Calculate the reward based on the actions taken. """
        # We settle our intraday imbalances - should be pretty low if agent is good.
        # if p_balancing < 0 we are buying unplanned electricity from the intraday market at 30% higher costs, else we are selling at 30% lower costs.
        balancing_reward = np.sum((p_balancing < 0) * p_balancing * prices * 1.3 + (p_balancing > 0) * p_balancing * prices * 0.7)
        
        # We calculate the cost of violating the technical limits of our plant with our chosen actions. Are normalized violations. 
        violation_cost = np.sum([np.sum(e**2) for e in technical_violations]) * self.technical_violation_cost  # Penalty for technical violations

        # Penalty for not meeting the hourly hydrogen contract (Value in €/tH2)
        h2_contract_penalty = np.sum([max(0, self.rfp.get_contract("Hydrogen1").parameters.get("penalty", 4000) * 
                                    (self.rfp.get_contract("Hydrogen1").parameters.get("volume", 0) - m_h2_to_pipeline[t])) for t in range(self.decision_horizon)])
        
        """ Check if the episode is done """
        self.time += pd.Timedelta(self.decision_horizon, 'hours')
        terminated = self.time >= self.year_end  # Terminate episode after one year of operations

        # Penalty for not meeting the yearly ammonia contract (Value in €/tNH3)
        nh3_contract_penalty = terminated * max(0, self.rfp.get_contract("Ammonia1").parameters.get("penalty", 1000) * 
                                    (self.rfp.get_contract("Ammonia1").parameters.get("volume", 0) - ytd_nh3))
        
        # Penalty for truncation violations equal to number of hours left in the year
        truncation_penalty = truncated * (self.year_end-self.time).total_seconds()/3600

        """ Summarize environment state at the end of the step in the info dict. """
        info = {}
        # We realize wind and solar now, in order to obtain VRE profiles for the next agent decision.
        info['system_solar_realization'], info['system_wind_realization'] = self.forecaster.realize_vre(start=self.time, end=self.time + pd.Timedelta(self.decision_horizon-1, 'h')) # DF
        info['asset_solar_realization'] = self.solar_mapper(info['system_solar_realization'])
        info['asset_wind_realization'] = self.wind_mapper(info['system_wind_realization'])

        info["action"] = action
        info["clipped_action"] = clipped_action
        info["technical_violation_cost"] = violation_cost
        info["el_spot_revenue"] = np.sum(p_to_grid * prices)
        info["balancing_revenue"] = balancing_reward
        info["h2_revenue"] = np.sum(m_h2_to_pipeline * self.rfp.get_contract("Hydrogen1").parameters.get("price", 2000))
        info["h2_penalty"] = h2_contract_penalty
        info["nh3_production"] = np.sum(m_nh3_to_storage)
        info["nh3_production_value"] = info["nh3_production"] * self.rfp.get_contract("Ammonia1").parameters.get("price", 500)
        info["nh3_shipment"] = m_nh3_to_ship
        info["nh3_sales"] = info["nh3_shipment"] * self.rfp.get_contract("Ammonia1").parameters.get("price", 500)
        info["nh3_penalty"] = nh3_contract_penalty
        info["truncation_penalty"] = truncation_penalty
        info["balancing"] = p_balancing
        info["available_power"] = procured_power
        info["wind_ppa"] = p_wind
        info["solar_ppa"] = p_solar
        info["base_ppa"] = p_base
        info["electricity_price"] = prices
        info["soc_h2"] = soc_h2_list
        info["soc_nh3"] = soc_nh3_list
        info["nh3_spot_sale_maxxed"] = nh3_spot_sale_adjustment > 0
        
        # We also save the current state in the info for possible flexible access.
        info["final_soc_NH3"] = soc_nh3
        info["final_soc_H2"] = soc_h2
        info["Ammonia1_produced_ytd"] = ytd_nh3
        info["time"] = self.time
        info["annual_contract_deadline"] = self.year_end

        # Monetary summaries
        info["daily_cash_flow"] = info["el_spot_revenue"] + info["balancing_revenue"] + info["h2_revenue"] - info["h2_penalty"] - info["nh3_penalty"]
        info["actual_cash_flow"] = info["daily_cash_flow"] + info["nh3_sales"]

        value_generation = info["daily_cash_flow"] + info["nh3_production_value"]

        # The reward can be custom-defined based on how we want to train the agent.
        reward = value_generation - info["technical_violation_cost"]

        if truncated: # Inform about infeasibility causing truncation
            info["technical_violation_message"] = "Could not handle the flows determined by the agent."

        return self.state, reward, terminated, truncated, info

    def get_power_and_price_step(self, start:pd.Timestamp):
        # Load realized solar and wind profiles.
        df_solar  = self.solar_mapper(self.forecaster.solar_realization_cf['solar']) * self.solar_capacity
        df_wind   = self.wind_mapper(self.forecaster.wind_realization_cf['wind']) * self.wind_capacity
        # Realize prices - updates the forecaster object by including the new realizations.
        df_prices = self.forecaster.realize_prices(start=start, end=start+pd.Timedelta(self.decision_horizon-1, 'hours'))
        # Save only arrays:
        prices = df_prices['price'].values
        wind   = df_wind.values
        solar  = df_solar.values
        base   = self.get_baseload_ppa_power()
        base_ppa = np.asarray([base] * len(prices))
        return wind, solar, base_ppa, prices

    def get_baseload_ppa_power(self):
        power = 0
        for k, v in self.rfp.get_ppas():
            if not(v.simulate_profile): # Also add the available baseload PPAs as available power.
                power += v.parameters.get('capacity', 100)
        return power

    def reset(self, *, seed: int | None = None, ):
        if self.normalize_step:
            return self._normalized_reset( seed=seed)
        else:
            return self._reset(seed=seed)

    def _reset(self, *, seed: int | None = None, ):
        """
        Reset the environment to its initial state.
        Returns:
            tuple: The initial state of the environment, and additional info.
        """
        self.forecaster = copy.deepcopy(self.original_forecaster)
        self.state      = self.state_space.low
        self.time       = self.forecaster.t_init
        self.year_end   = self.forecaster.t_init + relativedelta(years=+1) - pd.Timedelta(1, 'hour')
        self.seed       = seed
        info            = {'time': self.time, "Ammonia1_produced_ytd": 0, "final_soc_H2": 0, "final_soc_NH3": 0, 'annual_contract_deadline': self.year_end}
        info['system_solar_realization'], info['system_wind_realization'] = self.forecaster.realize_vre(start=self.time, end=self.time + pd.Timedelta(self.decision_horizon-1, 'h')) # DF
        info['asset_solar_realization'] = self.solar_mapper(info['system_solar_realization'])
        info['asset_wind_realization'] = self.wind_mapper(info['system_wind_realization'])
        return self.state, info
    
    def _normalized_reset(self, *, seed: int | None = None, ):
        """
        Reset the environment to its initial state and return a normalized state.
        Returns:
            tuple: The normalized initial state of the environment, and additional info.
        """
        state, info = self._reset(seed=seed)
        normalized_state = self.state_scaler.transform(state)
        return normalized_state, info

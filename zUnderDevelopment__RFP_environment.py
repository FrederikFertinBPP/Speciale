import gymnasium as gym
import numpy as np
from common_scripts.RFP_initialization import RenewableFuelPlant, create_rfp

class RFPEnv(gym.Env):

    def __init__(self, rfp:RenewableFuelPlant, normalize:bool=True):
        """
        Initialize the rfp environment.
        This environment simulates a hypothetical path planning problem.
        """
        self.horizon = 1 # Planning horizon in hours
        self.dt = 1  # Time step in hours
        self.rfp = rfp
        self.state = None  # Placeholder for the state of the environment
        self.normalized_state = None  # Placeholder for the normalized state of the environment
        self.time = None  # Placeholder for the current time in the environment
        self.technical_violation_cost = 1.0
        self.normalize_step = normalize
        # Define the action space
        # The action space consists of three actions:
        # 1. Amount of electricity to sell to the grid (GCP)
        # 2. Amount of hydrogen to sell through the H2 pipeline
        # 3. Amount of ammonia to store in the ammonia storage
        # The action space is defined as a Box with lower and upper bounds based on the capacities of the components
        self.action_names = ['Grid Power Sale [MW]', 'Electrolyzer Power [MW]', "H2 Pipeline [tH2/h]", "NH3 Production [tNH3]"]
        self.action_space = gym.spaces.Box(low = np.asarray([[-self.rfp.get_component("GCP").parameters['capacity'], 0, 0, 0]]*self.horizon,
                                                             dtype = np.float64),
                                           high = np.asarray([[self.rfp.get_component("GCP").parameters['capacity'],
                                                              self.rfp.get_component("Electrolyzer").parameters['capacity'],
                                                              self.rfp.get_component("Hydrogen Pipeline").parameters['capacity'],
                                                              self.rfp.get_component("Haber-Bosch Plant").parameters['capacity']]] * self.horizon,
                                                              dtype = np.float64),
                                                              shape=(self.horizon,len(self.action_names)),
                                                              dtype = np.float64)
        # Define the observation space
        self.state_names, s_low, s_high = [], [], []
        for name, comp in self.rfp.get_components().items():
            if comp.is_storage:
                self.state_names.append(name)
                s_high.append(comp.parameters["capacity"])
                s_low.append(0)
        for name, cont in self.rfp.get_contracts().items():
            if cont.frequency != "hourly": # Non-hourly contracts are a part of the observation space as this info is needed for the agent to make decisions
                self.state_names.append(name)
                s_high.append(cont.parameters['volume']) # Set large upper bound for non-hourly contracts
                s_low.append(0)
        
        # The observation space is defined as a Box with lower and upper bounds based on the capacities
        self.observation_space = gym.spaces.Box(low = np.asarray([s_low] * self.horizon),
                                                high = np.asarray([s_high] * self.horizon),
                                                shape=(self.horizon,len(self.state_names)),
                                                dtype = np.float64)

    def step(self, action):
        if self.normalize_step:
            return self._normalized_step(action)
        else:
            return self._step(action)

    def _step(self, action):
        """
        Perform a step in the environment with the given action.
        Args:
            action np.array: The actions to take.
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
                technical_violations.extend(np.array([violations]))
            else:
                technical_violations.extend(violations)
            return clipped, violations
        # Ensure the action is within the action space bounds
        clipped_action, action_technical_violations = _clip(action, self.action_space.low, self.action_space.high)
        # Get the stochastic inputs for the environment
        p_wind, p_solar, price_el = self.realize_stochasticity(steps=self.horizon)
        # Current state of the environment
        soc_h2, soc_nh3, ytd_nh3, = self.state
        # Actions taken by the agent
        # (1) Amount of electricity to sell to the grid (2) Power flowing to the electrolyzer
        # (3) Amount of hydrogen to sell through the H2 pipeline (4) Amount of ammonia to store
        p_to_grid, p_electrolyzer, m_h2_to_pipeline, m_nh3_to_storage = clipped_action

        soc_nh3, nh3_flow_adjustment = _clip(m_nh3_to_storage + soc_nh3, self.observation_space.low[1], self.observation_space.high[1])  # Update the state of charge of ammonia
        m_nh3_to_storage += nh3_flow_adjustment  # Adjust the flow of ammonia to storage based on the action taken
        
        kappa_hb = self.rfp.get_component("Haber-Bosch Plant").parameters["rate"]
        m_h2_to_hb = m_nh3_to_storage * kappa_hb  # Amount of hydrogen to send to the Haber-Bosch plant
        
        eta_electrolyzer = self.rfp.get_component("Electrolyzer").parameters["rate"]  # Efficiency of the electrolyzer [t H2/MWh]
        m_h2_from_electrolyzer = p_electrolyzer * eta_electrolyzer  # Amount of hydrogen produced by the electrolyzer
        m_h2_to_storage = m_h2_from_electrolyzer - m_h2_to_pipeline - m_h2_to_hb # Amount of hydrogen to store in the H2 storage
        soc_h2, h2_flow_adjustment = _clip(m_h2_to_storage + soc_h2, self.observation_space.low[0], self.observation_space.high[0])  # Update the state of charge of hydrogen
        m_h2_to_storage += h2_flow_adjustment  # Adjust the flow of hydrogen to storage if needed
        # If the hydrogen storage could not handle the flows determined, we first adjust the electrolyzer, then the pipeline, and finally the Haber-Bosch plant
        if np.abs(h2_flow_adjustment) > 0:
            p_electrolyzer, pipeline_adjustment = _clip((m_h2_from_electrolyzer + h2_flow_adjustment) / eta_electrolyzer, self.action_space.low[1], self.action_space.high[1])
            if np.abs(pipeline_adjustment) > 0:
                m_h2_to_pipeline, hb_adjustment = _clip(m_h2_to_pipeline + pipeline_adjustment * eta_electrolyzer, self.action_space.low[2], self.action_space.high[2])
                if np.abs(hb_adjustment) > 0:
                    m_nh3_to_storage, warn1 = _clip((m_h2_to_hb - hb_adjustment) / kappa_hb, self.action_space.low[3], self.action_space.high[3])
                    soc_nh3, warn2 = _clip(soc_nh3 - hb_adjustment / kappa_hb, self.observation_space.low[1], self.observation_space.high[1])
                    truncated = np.abs(warn1) > 0 or np.abs(warn2) > 0

        eta_hb = self.rfp.get_component("Haber-Bosch Plant").parameters["rate2"]  # Efficiency of the Haber-Bosch plant [MWh/t NH3]
        p_to_hb = m_nh3_to_storage * eta_hb  # Amount of electricity to send to the Haber-Bosch plant
        p_compressor = (m_h2_to_storage > 0) * m_h2_to_storage * self.rfp.get_component("Hydrogen Storage").parameters["storage"]["charge"]["rate"]  # Power consumed by the compressor to fill the H2 storage. No power used if the storage is discharged.
        p_balancing = p_wind + p_solar - p_to_grid - p_to_hb - p_compressor
        
        # Update the state of the environment based on the actions taken
        if self.time % (30*24) == 0 and self.time > 0:  # Every 30 days, a ship comes to pick up ammonia
            m_nh3_to_ship = min(self.rfp.get_component("Ammonia Shipment").parameters["capacity"], soc_nh3)  # Amount of ammonia to ship every 30 days
            soc_nh3 -= m_nh3_to_ship  # Update the state of charge of ammonia
            ytd_nh3 += m_nh3_to_ship  # Update the year-to-date ammonia delivery

        self.state = np.asarray([soc_h2, soc_nh3, ytd_nh3], dtype=np.float64)

        # Calculate the reward based on the actions taken
        if p_balancing < 0: # If we are buying unplanned electricity from the grid
            balancing_reward = p_balancing * price_el * 1.5 # Added cost for buying electricity
        else: # If we are selling unplanned electricity to the grid
            balancing_reward = p_balancing * price_el * 0.5 # Reduced reward for selling electricity to the grid
        violation_cost = np.sum(e**2 for e in technical_violations) * self.technical_violation_cost  # Penalty for technical violations

        h2_contract_penalty = max(0, # Penalty for not meeting the hydrogen contract
                                    self.rfp.get_contract("Hydrogen Offtake Agreement").parameters["penalty"] * 
                                    (self.rfp.get_contract("Hydrogen Offtake Agreement").parameters["volume"] - m_h2_to_pipeline))
        # Check if the episode is done
        self.time += self.dt
        terminated = self.time >= 8760  # End after one year (8760 hours)
        nh3_contract_penalty = terminated * max(0, # Penalty for not meeting the ammonia contract
                                    self.rfp.get_contract("Ammonia Offtake Agreement").parameters["penalty"] * 
                                    (self.rfp.get_contract("Ammonia Offtake Agreement").parameters["volume"] - ytd_nh3))
        truncation_penalty = truncated * (8760-self.time) * (1-terminated)  # Penalty for truncation violations

        info = {}
        info["action"] = action
        info["clipped_action"] = action
        info["technical_violation_cost"] = violation_cost
        info["el_spot_revenue"] = p_to_grid * price_el
        info["balancing_revenue"] = balancing_reward
        info["h2_revenue"] = m_h2_to_pipeline * self.rfp.get_contract("Hydrogen Offtake Agreement").parameters["price"]
        info["h2_penalty"] = h2_contract_penalty
        info["nh3_revenue"] = m_nh3_to_storage * self.rfp.get_contract("Ammonia Offtake Agreement").parameters["price"]
        info["nh3_penalty"] = nh3_contract_penalty
        info["truncation_penalty"] = truncation_penalty
        info["balancing"] = p_balancing
        info["wind_power"] = p_wind
        info["solar_power"] = p_solar
        info["electricity_price"] = price_el
        info["time"] = self.time
        info["hour_of_month"] = self.time % (30*24)

        reward_actual = info["el_spot_revenue"] + info["balancing_revenue"] + \
                    info["h2_revenue"] - info["h2_penalty"] + \
                        info["nh3_revenue"] - info["nh3_penalty"] - \
                            (info["technical_violation_cost"] + info["truncation_penalty"])  # Total reward calculation
        info["actual_reward"] = reward_actual

        reward = info["h2_revenue"] + info["nh3_revenue"] - info["technical_violation_cost"]


        if truncated:
            info["technical_violation_message"] = "Could not handle the flows determined by the agent."

        return self.state, reward, terminated, truncated, info

    def _normalized_step(self, normalized_action):
        """
        Perform a step in the environment with the given normalized action.
        Args:
            normalized_action (np.array): The normalized action in the range [0, 1].
        Returns:
            tuple: A tuple containing the next state, reward, terminated flag, truncated flag, and additional info.
        """
        # Denormalize the action to the original action space
        action = (normalized_action) * (self.action_space.high - self.action_space.low) + self.action_space.low
        state, reward, terminated, truncated, info = self._step(action)
        normalized_state = (state - self.observation_space.low) / (self.observation_space.high - self.observation_space.low)
        return normalized_state, reward, terminated, truncated, info

    def realize_stochasticity(self, steps=1):
        """
        Makes a one-step realization of the stochastic inputs for the environment.
        Returns:
            tuple: A tuple containing the realized wind power, solar power, and electricity price.
        """
        p_wind = np.random.uniform(0, self.rfp.get_component("WindPower").parameters["capacity"], size=steps)  # Simulated wind power in MW
        p_solar = np.random.uniform(0, self.rfp.get_component("SolarPower").parameters["capacity"], size=steps)  # Simulated solar power in MW
        price_el = np.random.uniform(0, 100, size=steps)  # Simulated electricity price in €/MWh
        return p_wind, p_solar, price_el

    def reset(self, *, seed: int | None = None):
        if self.normalize_step:
            return self._normalized_reset(seed=seed)
        else:
            return self._reset(seed=seed)

    def _reset(self,
        *,
        seed: int | None = None,):
        """
        Reset the environment to its initial state.
        Returns:
            tuple: The initial state of the environment, and additional info.
        """
        self.state = self.observation_space.low
        self.time = 0
        self.seed = seed

        return self.state, {}
    
    def _normalized_reset(self,
        *,
        seed: int | None = None,):
        """
        Reset the environment to its initial state and return a normalized state.
        Returns:
            tuple: The normalized initial state of the environment, and additional info.
        """
        state, info = self._reset(seed=seed)
        normalized_state = (state - self.observation_space.low) / (self.observation_space.high - self.observation_space.low)
        return normalized_state, info

def make_rfp_env(normalize:bool=True):
    """
    Create a hybrid power plant environment with predefined components and contracts.
    """
    rfp = create_rfp()
    env = RFPEnv(rfp, normalize=normalize)
    return env

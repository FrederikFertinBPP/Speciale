""" Necessary path addendum if we want to run this script not from the root. 
    To run from root call:
    python -m test_scripts.SCRIPTNAME """
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model_scripts.capacity_planning_extension import CapacityPlanningModel
from common_scripts.RFP_initialization import create_rfp
from common_scripts.utils import load_trajectories, load_stats
import numpy as np
import pandas as pd

import pyomo.environ as pyo

def get_data():
    experiment_name = "test_contract_DeterministicHA_production_value_ph_96_spot_True_small"
    experiment_name = "backcasting_persistence_DeterministicHA_production_value_ph_96_spot_True_small"
    trajectories = load_trajectories(experiment_name)
    trajectory = trajectories[-1] # Get the most recent trajectory from the experiment
    stats = load_stats(experiment_name)
    stats = stats[-1] # Get the most recent stats from the experiment

    time_index = pd.to_datetime(pd.date_range(start=trajectory.env_info[0]['time'], end=trajectory.env_info[-1]['time'], freq='h')[:-1], utc=True)
    in_sample_time = time_index.year % 2 == 1
    time_index = time_index[in_sample_time]

    wind_profile, solar_profile, electricity_price = [], [], []
    solar_cap, wind_cap = stats['solarpower_capacity'], stats['windpower_capacity']
    horizon_days = len(trajectory.reward)
    for t in range(1, horizon_days+1):
        wind_profile += list(trajectory.env_info[t]['ppa_power']['WindPower'] / wind_cap if wind_cap>0 else np.zeros(len(trajectory.env_info[t]['ppa_power']['WindPower'])))
        solar_profile += list(trajectory.env_info[t]['ppa_power']['SolarPower'] / solar_cap if solar_cap>0 else np.zeros(len(trajectory.env_info[t]['ppa_power']['SolarPower'])))
        electricity_price += list(trajectory.env_info[t]['electricity_price'])
    wind_profile = np.array(wind_profile)[in_sample_time]
    solar_profile = np.array(solar_profile)[in_sample_time]
    electricity_price = np.array(electricity_price)[in_sample_time]

    horizon = len(electricity_price)
    wind_cf = {('WindPower', t): wind_profile[t] for t in range(horizon)}
    solar_cf = {('SolarPower', t): solar_profile[t] for t in range(horizon)}
    nuclear_cf = {('NuclearPower', t): 1.0 for t in range(horizon)}
    supplier_cf = {**wind_cf, **solar_cf, **nuclear_cf,}
    electricity_price = {t: electricity_price[t] for t in range(horizon)}
    datetime_data = {t: time_index[t] for t in range(horizon)}
    data = {
        None: {
            'T_datetime' : datetime_data,
            'supplier_cf': supplier_cf,
            'electricity_price': electricity_price,
        }
    }
    return data, horizon

def main():
    solver = 'gurobi'
    allow_spot_buy = True
    rfp = create_rfp(layout_file="rfp_layout - resized.xlsx")

    data, horizon = get_data()

    capacity_planner = CapacityPlanningModel(rfp=rfp,
                                  planning_horizon=horizon, decision_horizon=horizon,
                                  solver=solver, allow_spot_buy=allow_spot_buy, inflexible=True,
                                  capacity_planning=True, discount_rate=0.08)
    capacity_planner.initialize_model()
    capacity_planner.build_concrete_instance(data=data)
    capacity_planner.run(verbose=True)
    print("Capacities optimized.")


import cProfile
if __name__ == '__main__':
    cProfile.run("main()", "run_profiles/capacity_planning.prof")
    # Example of how to read the profile results:
    import pstats
    prof = pstats.Stats("run_profiles/capacity_planning.prof")
    prof.strip_dirs().sort_stats("cumtime").print_stats(10)
    # cProfile.py -- Profile Python programs
""" Necessary path addendum if we want to run this script not from the root. 
    To run from root call:
    python -m test_scripts.SCRIPTNAME """
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model_scripts.capacity_planning_extension import CapacityPlanningModel
from model_scripts.environment import VRESystemToAssetMapping, EmissionFactorEstimator
from common_scripts.RFP_initialization import create_rfp
from common_scripts.utils import load_trajectories, load_stats, cache_read
from data_scripts.data_loader import HistoricalData

import numpy as np
import pandas as pd
from time import time

def get_data(rfp):
    t_s = time()
    start   = pd.Timestamp('20150101', tz='UTC')
    end     = pd.Timestamp('20251231', tz='UTC')
    data_object = HistoricalData(start=start, end=end, country_code='PT', server='ENTSOE', create_time_features=False)
    t_e = time()
    print(f"Data retrieval and preprocessing took {t_e-t_s:.2f} seconds.")
    df = data_object.data

    """ Retrieve mappers, which produce VRE profiles for single assets, given a system level production profile. """
    cache_path_mappers = os.getcwd() + "/models/plant_models/"
    vres = ["wind", "solar"]
    vre_mappers = {}
    for tag in vres:
        mapper = cache_read(cache_path_mappers + f"{tag}.pkl")
        vre_mappers[tag] = VRESystemToAssetMapping(mapper)

    ym  = df.index.tz_localize(None).to_period('M')
    for tag in vres:
        monthly_caps = ym.map(data_object.caps[tag])
        df[f"{tag}_cf"] = vre_mappers[tag](df[tag] / monthly_caps)
    
    # Calculates "Carbon intensity gCO₂eq/kWh (direct)" as a linear function of price [€/MWh], system wind [MW], and system solar [MW].
    cache_path_mappers = os.getcwd() + "/models/plant_models/"
    emissions_mapper = cache_read(cache_path_mappers + "emission_factor.pkl")
    emissions_model = EmissionFactorEstimator(emissions_mapper)

    X = df[["price", "solar", "wind", "Actual Load"]]
    df["emissions"] = emissions_model(X)

    insample_data = df.loc[df.index.year <= 2024]
    # insample_data = df.loc[(df.index.year >= 2023) & (df.index.year <= 2024)]
    horizon = len(insample_data)
    supplier_cf = {}
    for name, ppa in rfp.get_ppas().items():
        if ppa.parameters.get("consumes") == 'wind':
            cf = insample_data["wind_cf"].values
        elif ppa.parameters.get("consumes") == 'solar':
            cf = insample_data["solar_cf"].values
        else:
            cf = np.ones(horizon) # Assumes full availability of non-variable PPAs.
        supplier_cf.update({(name, t): cf[t] for t in range(horizon)})
    electricity_price = {t: insample_data["price"].iloc[t] for t in range(horizon)}
    emissions_intensity = {t: insample_data["emissions"].iloc[t] for t in range(horizon)}
    ets_price = {t: insample_data["ets"].iloc[t] for t in range(horizon)}
    datetime_data = {t: insample_data.index[t] for t in range(horizon)}
    data = {
        None: {
            'T_datetime' : datetime_data,
            'supplier_cf': supplier_cf,
            'electricity_price': electricity_price,
            'grid_emissions_intensity': emissions_intensity,
            'ets_price': ets_price,
        }
    }

    # PPA prices based on renewables market value the last 2 years of insample data (2023 and 2024).
    ppa_prices = {}
    df_ppa = df.loc[(df.index.year >= 2023) & (df.index.year <= 2024)]
    for tag in vres:
        ppa_prices[tag] = sum(df_ppa[f"{tag}_cf"] * df_ppa["price"]) / sum(df_ppa[f"{tag}_cf"])
    ppa_prices["baseload"] = np.mean(df_ppa["price"]) * 1.2 # Renewable baseload PPA is 20% above market price.

    return data, horizon, ppa_prices

def main():
    #%% Configuration:
    solver = 'gurobi'
    allow_spot_buy = True
    scenario_name = "" # Same as default
    layout_file = "article.xlsx"
    cvar_info = {"cvar_alpha":0.9, "cvar_beta":0.5}
    # cvar_info = None

    #%% Preprocessing

    rfp = create_rfp(scenario_name=scenario_name, layout_file=layout_file)
    if cvar_info is not None:
        cvar_str = str(cvar_info).replace(":","").replace("{","").replace("}","").replace("'","").replace(",","").replace(" ","_").replace(".","").replace("_cvar","")
    else:
        cvar_str = "risk_neutral"
    results_folder = f"setup_files/results/{rfp.layout_file.split(".")[0]}_{scenario_name}"
    os.makedirs(results_folder, exist_ok=True)

    data, horizon, ppa_prices = get_data(rfp)
    for name, ppa in rfp.get_ppas().items():
        resource = ppa.parameters.get("consumes")
        if resource in ('wind', 'solar'):
            price = ppa_prices[resource]
        else:
            price = ppa_prices["baseload"]
        ppa.parameters["price"] = np.round(price, 2)

    #%% Capacity Optimization
    capacity_planner = CapacityPlanningModel(rfp=rfp,
                                  planning_horizon=horizon, decision_horizon=horizon,
                                  enforce_rfnbo=True, inflexible=True,
                                  solver=solver, allow_spot_buy=allow_spot_buy, 
                                  capacity_planning=True, discount_rate=0.05,
                                  cvar_info=cvar_info,
                                  )
    capacity_planner.initialize_model()
    capacity_planner.build_concrete_instance(data=data)
    capacity_planner.run(verbose=True) # Completed the run with 10 years of data in 1603 seconds, full script in 1725 seconds. With CVaR, it was 2940 seconds run with 3066 seconds full script.
    print("Capacities optimized.")

    #%% Save main results:
    capacity_planner.save_optimal_capacities(f"{results_folder}/optimal_capacities-{cvar_str}.csv")
    capacity_planner.save_capacity_utilization_factors(f"{results_folder}/cf-{cvar_str}.csv")
    print("Results saved.")

#%% Timing
import cProfile
if __name__ == '__main__':
    cProfile.run("main()", "run_profiles/capacity_planning.prof")
    # Example of how to read the profile results:
    import pstats
    prof = pstats.Stats("run_profiles/capacity_planning.prof")
    prof.strip_dirs().sort_stats("cumtime").print_stats(10)
    # cProfile.py -- Profile Python programs
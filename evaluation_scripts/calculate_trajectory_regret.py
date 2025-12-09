""" Necessary path addendum if we want to run this script not from the root. 
    To run from root call:
    python -m test_scripts.SCRIPTNAME """
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common_scripts.utils import load_trajectories, load_stats
from common_scripts.RFP_initialization import create_rfp
from model_scripts_old.lp_deterministic import HourlyDeterministicLPModel
from model_scripts_old.RFP_operational_environment import RFPOperationalEnv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rfp = create_rfp()

def get_hindsight_solution(trajectory, stats, model_class=HourlyDeterministicLPModel):
    horizon_days = len(trajectory.reward)
    time_index = pd.to_datetime(pd.date_range(start=trajectory.env_info[0]['time'], end=trajectory.env_info[-1]['time'], freq='h')[:-1], utc=True)
    wind_profile = np.reshape([trajectory.env_info[t]['wind_ppa'] for t in range(1, horizon_days+1)],-1) / stats['wind_capacity']
    solar_profile = np.reshape([trajectory.env_info[t]['solar_ppa'] for t in range(1, horizon_days+1)],-1) / stats['solar_capacity']
    electricity_price = np.reshape([trajectory.env_info[t]['electricity_price'] for t in range(1, horizon_days+1)],-1)

    horizon = horizon_days * 24
    pfm = model_class(rfp,
                        planning_horizon=horizon,
                        solver='gurobi',
                        guideline='planning_target')
    pfm.initialize_model()
    wind_cf = {('WindPower', t): wind_profile[t] for t in range(horizon)}
    solar_cf = {('SolarPower', t): solar_profile[t] for t in range(horizon)}
    nuclear_cf = {('NuclearPower', t): 1.0 for t in range(horizon)}
    electricity_price = {t: electricity_price[t] for t in range(horizon)}
    data = {
        None: {
            'T_datetime' : {t: time_index[t] for t in range(horizon)},
            'init_soc': {
                'Hydrogen Storage': 0,
                'Ammonia Storage': 0,
            },
            'contract_target': {
                'Hydrogen1': rfp.get_contract("Hydrogen1").parameters.get('volume'),
                'Ammonia1': stats["contracted_yearly_targets"]['Ammonia1'].parameters.get('volume'),
            },
            'supplier_cf': {
                **wind_cf,
                **solar_cf,
                **nuclear_cf,
            },
            'electricity_price': electricity_price,
        }
    }
    pfm.build_concrete_instance(data=data)
    pfm.run()
    el_market_revenue = pfm.planning_results.exp_el_revenue
    return el_market_revenue, pfm

def get_realized_el_revenues(trajectory):
    horizon_days = len(trajectory.reward)
    el_da_revenue = np.sum([trajectory.env_info[t]['el_spot_cashflows'] for t in range(1, horizon_days+1)])
    el_ba_revenue = np.sum([trajectory.env_info[t]['balancing_revenue'] for t in range(1, horizon_days+1)])
    return el_da_revenue, el_ba_revenue

def get_realized_fuel_sale_revenues(trajectory, stats, normalized=False):
    horizon_days = len(trajectory.reward)
    h2_revenue = np.sum([trajectory.env_info[t]['h2_revenue'] for t in range(1, horizon_days+1)])
    nh3_produced_volume = np.sum([trajectory.env_info[t]['nh3_production'] for t in range(1, horizon_days+1)])
    nh3_contracted_volume = stats["contracted_yearly_targets"]['Ammonia1'].parameters.get('volume')
    nh3_contracted_sale = trajectory.state[-1][2] * (nh3_contracted_volume if normalized else 1)
    nh3_contract_revenue = rfp.get_contract('Ammonia1').parameters.get('price') * min(nh3_contracted_volume, nh3_contracted_sale)
    nh3_spot_revenue = rfp.get_contract('AmmoniaSpot').parameters.get('price') * max(0, nh3_produced_volume-min(nh3_contracted_volume, nh3_contracted_sale))
    return h2_revenue, nh3_contract_revenue, nh3_spot_revenue

def get_realized_costs(trajectory):
    horizon_days = len(trajectory.reward)
    wind_cost = np.sum([trajectory.env_info[t]['wind_ppa'] for t in range(1, horizon_days+1)]) * rfp.get_ppa('WindPower').parameters.get('price')
    solar_cost = np.sum([trajectory.env_info[t]['solar_ppa'] for t in range(1, horizon_days+1)]) * rfp.get_ppa('SolarPower').parameters.get('price')
    nuclear_cost = np.sum([trajectory.env_info[t]['base_ppa'] for t in range(1, horizon_days+1)]) * rfp.get_ppa('NuclearPower').parameters.get('price')
    return wind_cost, solar_cost, nuclear_cost

def print_trajectory_summary(trajectory, stats, model_class=HourlyDeterministicLPModel):
    normalized = stats.get("normalized",False)
    print("\n----------------------------------\n")
    print("Total reward: ", np.round(sum(trajectory.reward)*1e-6, 2), " M€")

    #%% Electricity revenue summary
    opt_el_revenue, opt_model = get_hindsight_solution(trajectory=trajectory, stats=stats, model_class=model_class)
    el_da_revenue, el_ba_revenue = get_realized_el_revenues(trajectory=trajectory)
    print("\n----------------------------------\n")
    print("Achieved electricity revenue (day-ahead): ", round(el_da_revenue))
    print("Achieved electricity revenue (balancing): ", round(el_ba_revenue))
    print("Achieved electricity revenue (total):\t",    round(el_ba_revenue+el_da_revenue))
    print("Optimal electricity revenue:\t\t",           round(opt_el_revenue))

    #%% Fuel sale summary
    h2_revenue, nh3_contract_revenue, nh3_spot_revenue = get_realized_fuel_sale_revenues(trajectory=trajectory, stats=stats, normalized=normalized)
    opt_h2_revenue = np.sum(opt_model.planning_results.fuel_sales['Hydrogen Pipeline']['Hydrogen1']) * rfp.get_contract('Hydrogen1').parameters.get('price')
    opt_nh3_contract_revenue = np.sum(opt_model.planning_results.fuel_sales['Ammonia Shipment']['Ammonia1']) * rfp.get_contract('Ammonia1').parameters.get('price')
    opt_nh3_spot_revenue = np.sum(opt_model.planning_results.fuel_sales['Ammonia Shipment']['AmmoniaSpot']) * rfp.get_contract('AmmoniaSpot').parameters.get('price')
    print("\n----------------------------------\n")
    contract_target = stats["contracted_yearly_targets"]['Ammonia1'].parameters.get('volume')
    print("Contracted annual ammonia:\t\t", round(contract_target), "t NH3")
    print("Annual production intended for contract:", round(trajectory.state[-1][2] * (contract_target if normalized else 1)), "t NH3")
    contract_shortfall  = trajectory.state[-1][2] * (contract_target if normalized else 1) - contract_target
    print("Deviation from required annual amount:\t", round(contract_shortfall), "t NH3")
    print("Achieved NH3 contract revenue:\t", round(nh3_contract_revenue))
    print("Achieved NH3 spot revenue:\t", round(nh3_spot_revenue))
    print("Optimal NH3 revenue:\t", round(opt_nh3_contract_revenue+opt_nh3_spot_revenue))
    print("Achieved H2 revenue:\t", round(h2_revenue))
    print("Optimal H2 revenue:\t", round(opt_h2_revenue))
    print("\n----------------------------------\n")
    realized_total_revenue = nh3_contract_revenue+nh3_spot_revenue+h2_revenue+el_da_revenue+el_ba_revenue
    print("Achieved total revenue:\t", round(realized_total_revenue))
    opt_total_revenue = opt_nh3_contract_revenue+opt_nh3_spot_revenue+opt_h2_revenue+opt_el_revenue
    print("Optimal total revenue:\t", round(opt_total_revenue))
    print("Percentage achieved: ", np.round(100 * realized_total_revenue / opt_total_revenue, 2), "%")

    #%% Cost summary
    print("\n----------------------------------\n")
    costs = get_realized_costs(trajectory=trajectory)
    print("Total PPA costs:\t", round(sum(costs)))
    contract_penalty = - 4 * min(contract_shortfall, 0) * rfp.get_contract('Ammonia1').parameters.get('price')
    print("Contract penalty (4 times the value):", round(contract_penalty))
    realized_profits = realized_total_revenue - sum(costs) - contract_penalty
    print("Total realized profits:\t", round(realized_profits))
    opt_profits = opt_total_revenue - sum(costs)
    print("Total optimal profits:\t", round(opt_profits))
    profit_percentage = 100 * realized_profits / opt_profits
    print("Percentage achieved: ", np.round(profit_percentage, 2), "%")

    #%% VRE availability summary
    horizon_days = len(trajectory.reward)
    wind_cf = np.mean([trajectory.env_info[t]['wind_ppa'] for t in range(1, horizon_days+1)]) / stats['wind_capacity']
    solar_cf = np.mean([trajectory.env_info[t]['solar_ppa'] for t in range(1, horizon_days+1)]) / stats['solar_capacity']
    print("\n----------------------------------\n")
    print("Solar capacity factor:\t", round(solar_cf*100, 3), "%")
    print("Wind capacity factor:\t", round(wind_cf*100, 3), "%")
    return profit_percentage, opt_model

if __name__ == '__main__':
    """ Change the experiment name to the one we want to evaluate: """
    experiment_name = "testStochastic_strike_price"
    experiment_name = "test3_strike_price"
    experiment_name = "testConstant"
    experiment_name = "testRL"
    # experiment_name = "test3_planning_target"
    experiment_name = "testContextAwareRL"
    experiment_name = "testStochastic_strike_price"

    #### We assess the regret of the model:
    # This is the difference in profits between the actions chosen by the agent and the optimal actions
    # chosen by an oracle: A perfect foresight model for the full year.
    # Just load the realized wind, solar, and prices (can be found in trajectory env info)
    # and solve an LP for the full year.

    #%% Trajectory stats:
    trajectories = load_trajectories(experiment_name)
    stats = load_stats(experiment_name, csv_version=False)
    print("\n----------------------------------\n")
    print("Experiment: ", experiment_name)

    profit_percentages = []
    opt_models = []
    for ix in range(len(trajectories)):
        pp, opt_model = print_trajectory_summary(trajectory=trajectories[ix], stats=stats[ix])
        profit_percentages.append(pp)
        opt_models.append(opt_model)
        print("Done")
    plt.hist(np.asarray(profit_percentages), label="Obtained profits")
    plt.axvline(np.mean(profit_percentages), label="Average obtained profit", lw=5, color='black')
    plt.xlabel(f"% of optimal profits")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(f"documentation/profit_dists/{experiment_name}.png")
    plt.close()
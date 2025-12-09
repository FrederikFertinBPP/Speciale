""" Necessary path addendum if we want to run this script not from the root. 
    To run from root call:
    python -m test_scripts.SCRIPTNAME """
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common_scripts.utils import load_trajectories, load_stats, cache_read
from common_scripts.RFP_initialization import create_rfp
from model_scripts.hourly_models import HourlyDeterministicLPModel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



# https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10839633&casa_token=-WRwu6R4PLkAAAAA:S8T2feRrOOVXyI0B3crc0-U9FuOidkP-GTvj-mnKkvcuyvo7C3hp5eNw-5UIxHJijxWzoJZz&tag=1

""" For each year we want the following numbers:
    Contracted Revenue, Electricity Revenue, PPA Costs, Electricity Costs, Power consumption, PPA power (contracted power), sold power (da+ba), bought power (da+ba).

    ^ We want to compute the following metrics:
    & Objective-oriented KPIs (computed for each scenario):
    EBITDA = Revenue - costs
    FixedRevenue = Contracted Revenue/(Contracted Revenue + Electricity Revenue)
    Cost Exposure = Electricity Costs/(PPA Costs + Electricity Costs)
    Revenue Exposure = Electricity Revenue/(Contracted Revenue + Electricity Revenue)
    Long exposure = Sold Power/Contracted Power
    Short exposure = Bought Power/Power consumption
    ? Gross exposure = (Sold Power + Bought Power)/Contracted Power

    & Uncertainty KPIs (evaluated across scenarios):
    P90 EBITDA vs expected EBITDA
    Likelihood of contract defaulting
"""

class EmissionFactorEstimator:
    """  """
    def __init__(self, model):
        self.model = model
    
    def __call__(self, *args, **kwds):
        return np.clip(self.model(*args, **kwds), 0, np.inf)

# Calculates "Carbon intensity gCO₂eq/kWh (direct)" as a linear function of price [€/MWh], system wind [MW], and system solar [MW].
cache_path_mappers = os.getcwd() + "/models/plant_models/"
mapper = cache_read(cache_path_mappers + "emission_factor.pkl")
emissions_model = EmissionFactorEstimator(mapper)

rfp = create_rfp()

def get_hindsight_solution(trajectory, stats, model_class=HourlyDeterministicLPModel):
    horizon_days = len(trajectory.reward)
    time_index = pd.to_datetime(pd.date_range(start=trajectory.env_info[0]['time'], end=trajectory.env_info[-1]['time'], freq='h')[:-1], utc=True)
    wind_profile, solar_profile, electricity_price = [], [], []
    for t in range(1, horizon_days+1):
        wind_profile += list(trajectory.env_info[t]['ppa_power']['WindPower'] / stats['windpower_capacity'])
        solar_profile += list(trajectory.env_info[t]['ppa_power']['SolarPower'] / stats['solarpower_capacity'])
        electricity_price += list(trajectory.env_info[t]['electricity_price'])

    horizon = horizon_days * 24
    allow_spot_buy = bool(stats["high"][0][0] > 0)
    pfm = model_class(rfp,
                        planning_horizon=horizon,
                        decision_horizon=horizon,
                        solver='gurobi',
                        allow_spot_buy=allow_spot_buy,
                        )
    pfm.initialize_model()
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
    pfm.build_concrete_instance(data=data)
    pfm.run()
    el_market_revenue = pfm.decision_results.exp_el_revenue
    return el_market_revenue, pfm

def get_realized_electricity_summary(trajectory, stats):
    horizon_days = len(trajectory.reward)
    T = range(1, horizon_days+1)
    if stats.get("balancing_market", False):
        el_revenue      = np.sum([trajectory.env_info[t]['el_revenue'] for t in T])
        el_da_bought    = np.sum([trajectory.env_info[t]['dayahead_bought'] for t in T])
        el_da_sold      = np.sum([trajectory.env_info[t]['dayahead_sold'] for t in T])
        el_ba_bought    = np.sum([trajectory.env_info[t]['balancing_power_bought'] for t in T])
        el_ba_sold      = np.sum([trajectory.env_info[t]['balancing_power_sold'] for t in T])
        el_ba_revenue   = -np.sum([trajectory.env_info[t]['balancing_cost'] for t in T])
        el_da_revenue   = -np.sum([trajectory.env_info[t]['dayahead_cost'] for t in T])
    else:
        el_revenue      = np.sum([trajectory.env_info[t]['el_spot_balance'] for t in T])
        el_da_revenue   = el_revenue
        el_ba_revenue   = 0
        el_da_bought    = np.sum([trajectory.env_info[t]['el_spot_bought'] for t in T])
        el_da_sold      = np.sum([trajectory.env_info[t]['el_spot_sold'] for t in T])
        el_ba_bought    = 0
        el_ba_sold      = 0
    el_bought = el_ba_bought + el_da_bought
    el_sold = el_ba_sold + el_da_sold
    summary = {"revenue": {"total": el_revenue, "dayahead": el_da_revenue, "balancing": el_ba_revenue},
               "volumes": {"bought": {"total": el_bought, "dayahead": el_da_bought, "balancing": el_ba_bought},
                           "sold": {"total": el_sold, "dayahead": el_da_sold, "balancing": el_ba_sold}}}
    return summary

def get_realized_fuel_sale_revenues(trajectory):
    horizon_days = len(trajectory.reward)
    T = range(1, horizon_days+1)
    contract_revenues = {cont: np.sum([trajectory.env_info[t]['contract_revenues'][cont] for t in range(1, horizon_days+1)]) for cont in trajectory.env_info[1]['contract_revenues'].keys()}
    contract_penalties = {cont: np.sum([sum(trajectory.env_info[t]['contract_penalties'][cont]) for t in range(1, horizon_days+1)]) for cont in trajectory.env_info[1]['contract_penalties'].keys()}
    return contract_revenues, contract_penalties

def get_realized_ppa_summary(trajectory, stats):
    horizon_days = len(trajectory.reward)
    T = range(1, horizon_days+1)
    wind_power = np.sum([sum(trajectory.env_info[t]['ppa_power']['WindPower']) for t in T])
    wind_cost = wind_power * rfp.get_ppa('WindPower').parameters.get('price')
    solar_power = np.sum([sum(trajectory.env_info[t]['ppa_power']['SolarPower']) for t in T])
    solar_cost = solar_power * rfp.get_ppa('SolarPower').parameters.get('price')
    nuclear_power = np.sum([sum(trajectory.env_info[t]['ppa_power']['NuclearPower']) for t in T])
    nuclear_cost = nuclear_power * rfp.get_ppa('NuclearPower').parameters.get('price')
    ppa_cost = sum([wind_cost, solar_cost, nuclear_cost])
    ppa_power = sum([wind_power, solar_power, nuclear_power])

    wind_cf, solar_cf = [], []
    for t in T:
        wind_cf += list(trajectory.env_info[t]['ppa_power']['WindPower'] / stats['windpower_capacity'])
        solar_cf += list(trajectory.env_info[t]['ppa_power']['SolarPower'] / stats['solarpower_capacity'])
    wind_cf = np.mean(wind_cf)
    solar_cf = np.mean(solar_cf)

    return ppa_cost, ppa_power, wind_cf, solar_cf

def print_trajectory_summary(trajectory, stats, model_class=HourlyDeterministicLPModel):
    normalized = stats.get("normalized",False)
    print("\n----------------------------------\n")
    print("Total reward: ", np.round(sum(trajectory.reward)*1e-6, 2), " M€")

    #%% Electricity revenue summary
    opt_el_revenue, opt_model = get_hindsight_solution(trajectory=trajectory, stats=stats, model_class=model_class)
    el_summary = get_realized_electricity_summary(trajectory=trajectory, stats=stats)
    print("\n----------------------------------\n")
    print("Achieved electricity revenue:\t", round(el_summary["revenue"]["total"]))
    print("Optimal electricity revenue:\t", round(opt_el_revenue))

    #%% Fuel sale summary
    contract_revenues, contract_penalties = get_realized_fuel_sale_revenues(trajectory=trajectory)
    opt_contract_revenues = opt_model.planning_results.delivered_revenue
    opt_contract_penalties = {cont : np.sum(opt_model.planning_results.contract_penalty[cont]) for cont in trajectory.env_info[1]['contract_penalties']}
    print("\n----------------------------------\n")
    contract_target = stats['Ammonia1 volume']
    print("Contracted annual ammonia:\t\t", round(contract_target), "t NH3")
    prod_ammonia1 = trajectory.state[-1]['state']['contracts'][1] * (contract_target if normalized else 1)
    print("Annual production intended for contract:", round(prod_ammonia1), "t NH3")
    contract_shortfall  = prod_ammonia1 - contract_target
    print("Deviation from required annual amount:\t", round(contract_shortfall), "t NH3")
    print("Achieved contract revenues:\t", contract_revenues)
    print("Received penalties:\t", contract_penalties)
    print("Optimal contract revenues:\t", opt_contract_revenues)
    print("Optimal contract penalties:\t", opt_contract_penalties)
    print("\n----------------------------------\n")
    realized_total_revenue = sum(contract_revenues.values()) + el_summary["revenue"]["total"]
    print("Achieved total revenue:\t", round(realized_total_revenue))
    opt_total_revenue = sum(opt_contract_revenues.values())+opt_el_revenue
    print("Optimal total revenue:\t", round(opt_total_revenue))
    print("Percentage achieved: ", np.round(100 * realized_total_revenue / opt_total_revenue, 2), "%")

    #%% Cost summary
    print("\n----------------------------------\n")
    ppa_costs, ppa_power, wind_cf, solar_cf = get_realized_ppa_summary(trajectory=trajectory, stats=stats)
    print("Total PPA costs:\t", round(ppa_costs))
    realized_profits = realized_total_revenue - ppa_costs - sum(contract_penalties.values())
    print("Total realized profits:\t", round(realized_profits))
    optimal_profits = opt_total_revenue - ppa_costs
    print("Total optimal profits:\t", round(optimal_profits))
    profit_percentage = 100 * realized_profits / optimal_profits
    print("Percentage achieved: ", np.round(profit_percentage, 2), "%")

    #%% VRE availability summary
    print("\n----------------------------------\n")
    print("Solar capacity factor:\t", round(solar_cf*100, 3), "%")
    print("Wind capacity factor:\t", round(wind_cf*100, 3), "%")

    ebitda = realized_profits
    contracted_revenue = sum(contract_revenues.values()) - sum(contract_penalties.values())
    contracted_revenue_share = contracted_revenue / (contracted_revenue + el_summary["revenue"]["total"])
    short_exposure = el_summary["volumes"]["sold"]["total"] / ppa_power
    power_consumption = el_summary["volumes"]["bought"]["total"] + ppa_power - el_summary["volumes"]["sold"]["total"]
    long_exposure = el_summary["volumes"]["bought"]["total"] / power_consumption
    balancing_exposure = (el_summary["volumes"]["bought"]["balancing"] + el_summary["volumes"]["sold"]["balancing"]) / (el_summary["volumes"]["bought"]["total"] + el_summary["volumes"]["sold"]["total"])

    return (opt_model, optimal_profits, profit_percentage, realized_profits,
            ebitda, contracted_revenue, contracted_revenue_share, short_exposure, long_exposure,
            power_consumption, balancing_exposure)

if __name__ == '__main__':
    """ Change the experiment name to the one we want to evaluate: """
    experiment_name = "testNewEnv_correctShipment"
    experiment_name = "testNewEnvproduction_value"
    experiment_name = "testDecisionRuleTrain"
    experiment_name = "testStochasticSpotbuy_strike_price"
    experiment_name = "testDecisionRuleAgent"
    experiment_name = "testNewEnvStochasticproduction_value"
    experiment_name = "testDomainsDecisionRuleAgent"
    experiment_name = "test_DeterministicHA_hourly_target_ph_96_spot_True"
    experiment_name = "test_DeterministicHA_production_value_ph_96_spot_True"
    experiment_name = "test_RecourseAgent1_production_value_ph_96_spot_True"
    # experiment_name = "test_StochasticHA5_production_value_ph_96_spot_True"
    # experiment_name = "test_RecourseAgent5_production_value_ph_96_spot_True"
    # experiment_name = "testNewEnvRLproduction_valuevalue_maximization"
    # experiment_name = "testStrikePriceBidder"

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

    optimal_models = []
    optimal_profits = []
    profit_percentages = []
    realized_profits = []
    ebitdas = []
    contracted_revenues = []
    contracted_revenue_shares = []
    short_exposures = []
    long_exposures = []
    power_consumptions = []
    balancing_exposures = []
    runtimes = []
    for ix in range(len(trajectories)):
        om, op, pp, rp, ebitda, cr, crs, se, le, pc, be = print_trajectory_summary(trajectory=trajectories[ix], stats=stats[ix])
        optimal_models.append(om)
        optimal_profits.append(op)
        profit_percentages.append(pp)
        realized_profits.append(rp)
        ebitdas.append(ebitda)
        contracted_revenues.append(cr)
        contracted_revenue_shares.append(crs)
        short_exposures.append(se)
        long_exposures.append(le)
        power_consumptions.append(pc)
        balancing_exposures.append(be)
        runtimes.append(trajectories[ix].env_info[-1].get("episode_runtime",0))
        print(f"{ix} done")
    df = pd.DataFrame(data={"Optimal Profit":optimal_profits,
                        "Realized Profit":realized_profits,
                        "Percentage Obtained":profit_percentages,
                        "EBITDA":ebitdas,
                        "Contracted Revenue": contracted_revenues,
                        "Contracted Revenue Share": contracted_revenue_shares,
                        "Short Exposure": short_exposures,
                        "Long Exposure": long_exposures,
                        "Power Consumption": power_consumptions,
                        "Balancing Exposure": balancing_exposures,
                        "Episode Runtimes": runtimes,
                        })
    df.to_csv(f"evaluation_scripts/processed_results/{experiment_name}.csv")

    

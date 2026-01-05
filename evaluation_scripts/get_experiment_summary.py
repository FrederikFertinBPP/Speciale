""" Necessary path addendum if we want to run this script not from the root.
    To run from root call:
    python -m test_scripts.SCRIPTNAME """
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common_scripts.utils import load_trajectories, load_stats, cache_read
from common_scripts.RFP_initialization import create_rfp
from model_scripts.hourly_models import HourlyDeterministicLPModel
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme("notebook", font_scale=1.5, style="darkgrid")
plt.rcParams['font.size'] = 16
# set legend fontsize to 14
plt.rcParams['legend.fontsize'] = 18
# set the font weight of the legend to bold
plt.rcParams['legend.title_fontsize'] = 18
# set the font size of the x and y labels to 14
plt.rcParams['axes.labelsize'] = 18
# set the font weight of the x and y labels to bold
plt.rcParams['axes.labelweight'] = 'bold'
# set the font size of the x and y ticks to 12
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
# set the font size of the title to 16
plt.rcParams['axes.titlesize'] = 18
# set the font weight of the title to bold
plt.rcParams['axes.titleweight'] = 'bold'

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
    return pfm

def get_realization_summary(trajectory, stats):
    horizon_days = len(trajectory.reward)
    T = range(1, horizon_days+1)

    ### * Spot Market Trading Summaries * ###
    # Total revenue:
    el_revenue      = np.sum([trajectory.env_info[t]['el_revenue'] for t in T])
    
    # Detailed streams (Day-Ahead):
    da_prices, da_buy, da_sell = [], [], []
    for t in T:
        da_prices += list(trajectory.env_info[t]['electricity_price'])
        da_buy  += list(trajectory.env_info[t]['dayahead_buy_profile'])
        da_sell += list(trajectory.env_info[t]['dayahead_sell_profile'])
    da_prices, da_buy, da_sell = np.asarray(da_prices), np.asarray(da_buy), np.asarray(da_sell)
    
    el_da_bought = np.sum(da_buy)
    el_da_sold = np.sum(da_sell)
    el_da_revenue = np.sum(da_prices * (da_sell - da_buy))

    if stats.get("balancing_market", False):
        # If our agent is participating in the intraday market then we also evaluate this:
        ba_buy, ba_sell, ba_buy_prices, ba_sell_prices = [], [], [], []
        for t in T:
            ba_buy += list(trajectory.env_info[t]['balancing_buy_profile'])
            ba_sell += list(trajectory.env_info[t]['balancing_sell_profile'])
            ba_buy_prices += list(trajectory.env_info[t]['balancing_buy_prices'])
            ba_sell_prices += list(trajectory.env_info[t]['balancing_sell_prices'])
        ba_buy, ba_sell, ba_buy_prices, ba_sell_prices = np.asarray(ba_buy), np.asarray(ba_sell), np.asarray(ba_buy_prices), np.asarray(ba_sell_prices)

        el_ba_bought = np.sum(ba_buy)
        el_ba_sold = np.sum(ba_sell)
        el_ba_revenue = np.sum(ba_sell * ba_sell_prices) - np.sum(ba_buy * ba_buy_prices)
    else:
        ba_buy  = 0
        ba_sell = 0
        el_ba_revenue   = 0
        el_ba_bought    = 0
        el_ba_sold      = 0
    buy_profile = da_buy + ba_buy
    sell_profile = da_sell + ba_sell
    el_bought = el_ba_bought + el_da_bought
    el_sold = el_ba_sold + el_da_sold

    spot_summary = {"revenue": {"total": el_revenue, "dayahead": el_da_revenue, "balancing": el_ba_revenue},
                    "volumes": {"bought": {"total": el_bought, "dayahead": el_da_bought, "balancing": el_ba_bought},
                                "sold": {"total": el_sold, "dayahead": el_da_sold, "balancing": el_ba_sold}}}

    ### * Capture Price Overview * ###
    capture_price_summary = {}
    capture_price_summary["baseload"] = np.mean(da_prices)
    capture_price_summary["buy"], capture_price_summary["sell"] = {}, {}
    capture_price_summary["buy"]["dayahead"]  = np.sum(da_buy * da_prices) / el_da_bought
    capture_price_summary["sell"]["dayahead"] = np.sum(da_sell * da_prices) / el_da_sold
    if stats.get("balancing_market", False):
        capture_price_summary["buy"]["balancing"]  = np.sum(ba_buy * ba_buy_prices) / el_ba_bought
        capture_price_summary["sell"]["balancing"] = np.sum(ba_sell * ba_sell_prices) / el_ba_sold

    ### * Emissions overview * ###
    grid_emission_factor = []
    for t in T:
        grid_emission_factor += list(trajectory.env_info[t]['electricity_emissions'])
    grid_emission_factor = np.asarray(grid_emission_factor)

    emissions_summary = {}
    emissions_summary["average"] = np.mean(grid_emission_factor)
    emissions_summary["totals"] = {}
    emissions_summary["totals"]["buying"] = np.sum(buy_profile*grid_emission_factor)
    emissions_summary["totals"]["selling"] = np.sum(sell_profile*grid_emission_factor)

    emissions_summary["averages"] = {}
    emissions_summary["averages"]["buying"] = emissions_summary["totals"]["buying"] / el_bought
    emissions_summary["averages"]["selling"] = emissions_summary["totals"]["selling"] / el_sold
    emissions_summary["averages"]["selling"] = emissions_summary["totals"]["selling"] / el_sold

    ### * PPA Power Overview * ###
    ppa_profile = {}
    ppas = stats['ppas'].keys()
    for key in ppas:
        l = []
        for t in T:
            l += list(trajectory.env_info[t]['ppa_power'][key])
        ppa_profile[key] = np.asarray(l)
    ppa_production = {key: np.sum(ppa_profile[key]) for key in ppas}
    ppa_prices = {key: stats['ppas'][key].parameters.get('price') for key in ppas}
    ppa_cost = {key: ppa_production[key] * ppa_prices[key] for key in ppas}
    ppa_capacities = {key: stats['ppas'][key].parameters.get('capacity') for key in ppas}
    ppa_capacity_factor = {key: np.mean(ppa_profile[key]) / ppa_capacities[key] for key in ppas}
    ppa_capture_prices = {key: np.sum(ppa_profile[key] * da_prices) / ppa_production[key] for key in ppas}
    capture_price_summary["ppa"] = ppa_capture_prices
    emissions_summary["totals"]["baseload_ppa"] = emissions_summary["average"] * sum(ppa_production[key] for key in ppas if not(stats['ppas'][key].parameters.get("simulated")))
    emissions_summary["balance"] = emissions_summary["totals"]["buying"] + emissions_summary["totals"]["baseload_ppa"] - emissions_summary["totals"]["selling"]

    power_consumption = np.sum([sum(trajectory.env_info[t]['power_consumption']) for t in T])
    power_contracted = sum(ppa_production.values())

    hydrogen_produced = np.sum([sum(trajectory.env_info[t]['link_productions']['Electrolyzer']) for t in T])

    revenue_exposure = el_sold / (el_sold + power_consumption)
    cost_exposure = el_bought / (el_bought + power_contracted)

    short_exposure = el_bought / power_consumption
    long_exposure = el_sold / power_contracted

    balancing_exposure = (el_ba_bought + el_ba_sold) / (el_bought + el_sold)

    ### * Offtaker Contract Overview * ###
    contract_revenues = {cont: np.sum([trajectory.env_info[t]['contract_revenues'][cont] for t in range(1, horizon_days+1)]) for cont in trajectory.env_info[1]['contract_revenues'].keys()}
    contract_penalties = {cont: np.sum([sum(trajectory.env_info[t]['contract_penalties'][cont]) for t in range(1, horizon_days+1)]) for cont in trajectory.env_info[1]['contract_penalties'].keys()}
    contracted_revenue = sum(contract_revenues.values())
    contract_penalty = sum(contract_penalties.values())

    realized_total_revenue = contracted_revenue + spot_summary["revenue"]["total"]

    return (spot_summary, capture_price_summary, emissions_summary, ppa_capacity_factor, contract_revenues, contract_penalties,
            ppa_cost, el_revenue, contracted_revenue, contract_penalty,
            revenue_exposure, cost_exposure, short_exposure, long_exposure, balancing_exposure,
            emissions_summary["balance"], realized_total_revenue, hydrogen_produced)

def print_trajectory_summary(trajectory, stats, model_class = HourlyDeterministicLPModel, opt_model=None):
    normalized = stats.get("normalized",False)
    print("\n----------------------------------\n")
    print("Total reward: ", np.round(sum(trajectory.reward)*1e-6, 2), " M€")

    (spot_summary, capture_price_summary, emissions_summary, ppa_capacity_factor, contract_revenues, contract_penalties,
            ppa_cost, el_revenue, contracted_revenue, contract_penalty,
            revenue_exposure, cost_exposure, short_exposure, long_exposure, balancing_exposure,
            emissions_balance, realized_total_revenue, hydrogen_produced) = get_realization_summary(trajectory=trajectory, stats=stats)

    #%% Electricity revenue summary
    if opt_model is None:
        opt_model = get_hindsight_solution(trajectory=trajectory, stats=stats, model_class=model_class)
    opt_el_revenue = opt_model.decision_results.exp_el_revenue
    print("\n----------------------------------\n")
    print("Achieved electricity revenue:\t", round(spot_summary["revenue"]["total"]))
    print("Optimal electricity revenue:\t", round(opt_el_revenue))

    #%% Fuel sale summary
    opt_contract_revenues = opt_model.planning_results.delivered_revenue
    opt_contract_penalties = {cont : np.sum(opt_model.planning_results.contract_penalty[cont]) for cont in trajectory.env_info[1]['contract_penalties']}
    print("\n----------------------------------\n")
    contract_target = stats['Ammonia1 volume']
    print("Contracted annual ammonia:\t\t", round(contract_target), "t NH3")
    prod_ammonia1 = trajectory.state[-1]['state']['contracts'][1] * (contract_target if normalized else 1)
    print("Annual production intended for contract:", round(prod_ammonia1), "t NH3")
    contract_shortfall  = prod_ammonia1 - contract_target
    print("Deviation from required annual amount:\t", round(contract_shortfall), "t NH3")
    print("Achieved contract revenues:\t", contracted_revenue)
    print("Received penalties:\t", contract_penalties)
    print("Optimal contract revenues:\t", opt_contract_revenues)
    print("Optimal contract penalties:\t", opt_contract_penalties)
    print("\n----------------------------------\n")
    
    print("Achieved total revenue:\t", round(realized_total_revenue))
    opt_total_revenue = sum(opt_contract_revenues.values())+opt_el_revenue
    print("Optimal total revenue:\t", round(opt_total_revenue))
    print("Percentage achieved: ", np.round(100 * realized_total_revenue / opt_total_revenue, 2), "%")

    #%% Cost summary
    print("\n----------------------------------\n")
    tot_ppa_costs = sum(ppa_cost.values())
    print("Total PPA costs:\t", round(tot_ppa_costs))
    realized_profits = realized_total_revenue - tot_ppa_costs - sum(contract_penalties.values())
    print("Total realized profits:\t", round(realized_profits))
    optimal_profits = opt_total_revenue - tot_ppa_costs
    print("Total optimal profits:\t", round(optimal_profits))
    profit_percentage = 100 * realized_profits / optimal_profits
    print("Percentage achieved: ", np.round(profit_percentage, 2), "%")

    #%% VRE availability summary
    print("\n----------------------------------\n")
    print("PPA capacity factors: ", ppa_capacity_factor)

    ebitda = realized_profits
    contracted_revenue -= contract_penalty
    contracted_revenue_share = contracted_revenue / (contracted_revenue + spot_summary["revenue"]["total"])
    runtime = trajectories[ix].env_info[-1].get("episode_runtime",0)

    trajectory_summary = {"Optimal Profit [€]": optimal_profits, "Contract Defaulting [€]": contract_penalty,
                        "Profit Percentage [%]": profit_percentage, "EBITDA [€]": ebitda,
                        "Short Exposure [%]": short_exposure, "Long Exposure [%]": long_exposure, 
                        "Revenue Exposure [%]": revenue_exposure, "Cost Exposure [%]": cost_exposure, 
                        "Balancing Exposure [%]": balancing_exposure, "Scope 2 Emissions [tCO2]": emissions_balance,
                        "Hydrogen Produced [tH2]": hydrogen_produced, "Episode Runtime [s]": runtime,
                        }

    return opt_model, trajectory_summary, emissions_summary, capture_price_summary, ppa_capacity_factor


if __name__ == '__main__':
    documentation = False
    single_experiment_evaluation = False
    document_optimal_strategy = False

    """ Change the experiment name to the one we want to evaluate: """
    experiment_name = "planningsensitivity_DeterministicHA_production_value_ph_96_spot_True"

    experiments = ("test_DeterministicHA_hourly_target_ph_96_spot_True",
                   "test_DeterministicHA_production_value_ph_96_spot_True",
                   "test_StochasticHA5_production_value_ph_96_spot_True",
                   "test_RecourseAgent1_production_value_ph_96_spot_True",
                   "test_RecourseAgent5_production_value_ph_96_spot_True",
                   "test_RecourseAgent5_DAbidding_production_value_ph_96_spot_True",
                   "test_StrikePriceBiddingAgent1_SP1_production_value_ph_96_spot_True",
                   "test_StrikePriceBiddingAgent1_SP5_production_value_ph_96_spot_True",
                   "test_BiddingCurveAgent1_D1_ph_96_spot_True",
                   "test_BiddingCurveAgent1_D2_ph_96_spot_True",
                   "test_BiddingCurveAgent1_D3_ph_96_spot_True",
                )
    # experiments = ("planningsensitivity_DeterministicHA_production_value_ph_24_spot_True", 
    #                "planningsensitivity_DeterministicHA_production_value_ph_48_spot_True", 
    #                "planningsensitivity_DeterministicHA_production_value_ph_72_spot_True", 
    #                "planningsensitivity_DeterministicHA_production_value_ph_96_spot_True", 
    #                )
    experiments = ("test_RecourseAgent5_production_value_ph_96_spot_True",
                   "test_RecourseAgent5_DAbidding_production_value_ph_96_spot_True",
                )
    #### We assess the regret of the model:
    # This is the difference in profits between the actions chosen by the agent and the optimal actions
    # chosen by an oracle: A perfect foresight model for the full year.
    # Just load the realized wind, solar, and prices (can be found in trajectory env info)
    # and solve an LP for the full year.

    #%% Trajectory stats:
    if single_experiment_evaluation:
        trajectories = load_trajectories(experiment_name)
        stats = load_stats(experiment_name, csv_version=False)
        print("\n----------------------------------\n")
        print("Experiment: ", experiment_name)
        
        # Initialize DataFrames for each summary
        trajectory_summaries = []
        emissions_summaries = []
        capture_price_summaries = []
        ppa_capacity_factors = []

        for ix in range(len(trajectories)):
            opt_model, trajectory_summary, emissions_summary, capture_price_summary, ppa_capacity_factor = print_trajectory_summary(trajectory=trajectories[ix], stats=stats[ix])
            trajectory_summaries.append(trajectory_summary)
            emissions_summaries.append(emissions_summary)
            capture_price_summaries.append(capture_price_summary)
            ppa_capacity_factors.append(ppa_capacity_factor)
            print(f"{experiment_name}'s trajectory number {ix} done")
        os.makedirs(f"evaluation_scripts/processed_results/{experiment_name}", exist_ok=True)
        pd.DataFrame(trajectory_summaries).to_csv(f"evaluation_scripts/processed_results/{experiment_name}/trajectory_summary.csv", index=False)
        pd.DataFrame(emissions_summaries).to_csv(f"evaluation_scripts/processed_results/{experiment_name}/emissions_summary.csv", index=False)
        pd.DataFrame(capture_price_summaries).to_csv(f"evaluation_scripts/processed_results/{experiment_name}/capture_price_summary.csv", index=False)
        pd.DataFrame(ppa_capacity_factors).to_csv(f"evaluation_scripts/processed_results/{experiment_name}/ppa_capacity_factors.csv", index=False)
        print(f"Evaluation of {experiment_name} finished")
    else:
        opt_models = []
        for exp_ix, experiment_name in enumerate(experiments):
            trajectories = load_trajectories(experiment_name)
            stats = load_stats(experiment_name, csv_version=False)

            trajectory_summaries = []
            emissions_summaries = []
            capture_price_summaries = []
            ppa_capacity_factors = []
            
            for ix in range(len(trajectories)):
                if exp_ix == 0:
                    opt_model, trajectory_summary, emissions_summary, capture_price_summary, ppa_capacity_factor = print_trajectory_summary(trajectory=trajectories[ix], stats=stats[ix])
                    opt_models.append(opt_model)
                    if documentation:
                        prices = [p.value for p in opt_model.inst.electricity_price.values()]
                        imports = list(opt_model.decision_results.link_production["Grid Connection Point"])
                        fig, ax = plt.subplots(figsize=(12,8))
                        plt.scatter(prices, imports, s=2)
                        plt.xlabel("Prices [€/MWh]")
                        plt.ylabel("Net Power Consumption [MW]")
                        plt.savefig(f"documentation/PerfectForesightStrategy/scatter_{ix}.png")
                        plt.show()
                        ss = [[p,i] for p,i in zip(prices, imports)]
                        sort_ss = sorted(ss) # Sort by prices, index 0
                        data = np.asarray(sort_ss)
                        fig, ax = plt.subplots(figsize=(12,8))
                        for ix, datapoint in enumerate(data):
                            color= "steelblue" if datapoint[1]>1 else "lightgrey"
                            plt.bar(ix, datapoint[0], width=0.1, color=color, edgecolor=color)
                        plt.bar(0,0, width=0.1, color="steelblue", label="Consuming")
                        plt.bar(0,0, width=0.1, color="lightgrey", label="Not Consuming")
                        plt.xlabel("Sorted hours")
                        plt.ylabel("Electricity Price [€/MWh]")
                        plt.legend()
                        plt.savefig(f"documentation/PerfectForesightStrategy/duration_curve_{ix}.png")
                        plt.show()
                else:
                    _, trajectory_summary, emissions_summary, capture_price_summary, ppa_capacity_factor = print_trajectory_summary(trajectory=trajectories[ix], stats=stats[ix], opt_model=opt_models[ix])
                print(f"{experiment_name}'s trajectory number {ix} done")
                trajectory_summaries.append(trajectory_summary)
                emissions_summaries.append(emissions_summary)
                capture_price_summaries.append(capture_price_summary)
                ppa_capacity_factors.append(ppa_capacity_factor)
                print(f"{experiment_name}'s trajectory number {ix} done")
            # Make sure the output directory exists
            os.makedirs(f"evaluation_scripts/processed_results/{experiment_name}", exist_ok=True)
            pd.DataFrame(trajectory_summaries).to_csv(f"evaluation_scripts/processed_results/{experiment_name}/trajectory_summary.csv", index=False)
            pd.DataFrame(emissions_summaries).to_csv(f"evaluation_scripts/processed_results/{experiment_name}/emissions_summary.csv", index=False)
            pd.DataFrame(capture_price_summaries).to_csv(f"evaluation_scripts/processed_results/{experiment_name}/capture_price_summary.csv", index=False)
            pd.DataFrame(ppa_capacity_factors).to_csv(f"evaluation_scripts/processed_results/{experiment_name}/ppa_capacity_factors.csv", index=False)
            print(f"Evaluation of {experiment_name} finished")



    

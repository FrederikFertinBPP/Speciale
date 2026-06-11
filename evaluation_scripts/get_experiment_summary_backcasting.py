""" Necessary path addendum if we want to run this script not from the root.
    To run from root call:
    python -m test_scripts.SCRIPTNAME """
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common_scripts.utils import load_trajectories, load_stats, Trajectory
from common_scripts.RFP_initialization import create_rfp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common_scripts.utils import set_plotting_style
set_plotting_style()
from evaluation_scripts.get_experiment_summary import print_trajectory_summary


def get_yearly_summary(trajectory, stats, rfp, year, opt_model = None):
    day_times = np.asarray([trajectory.env_info[x]["time"].year for x in range(len(trajectory.reward))])
    slice_indices = np.where(day_times == year)[0]
    traj_slice = Trajectory(state=list(np.asarray(trajectory.state)[slice_indices]) + [trajectory.state[max(slice_indices)+1]],
                            action=list(np.asarray(trajectory.action)[slice_indices]),
                            reward=list(np.asarray(trajectory.reward)[slice_indices]),
                            time=list(np.asarray(trajectory.time)[slice_indices]),
                            env_info=list(np.asarray(trajectory.env_info)[slice_indices]) + [trajectory.env_info[max(slice_indices)+1]],
    )
    return print_trajectory_summary(trajectory=traj_slice, stats=stats, rfp=rfp, opt_model=opt_model, plot_emissions=True)

if __name__ == '__main__':
    documentation = False # "Backcasting"
    single_experiment_evaluation = False
    document_optimal_strategy = False
    layout_file="article.xlsx"
    scenario_name="100"

    experiments = ("backcasting_AggregateFullHorizonAgent_ph_96_spot_True_out_of_sample",
                   "backcasting_DeterministicHA_production_value_ph_96_spot_True_out_of_sample")
    agents = ("AggregateFullHorizonAgent_ph_96_spot_True_out_of_sample",
                "DeterministicHA_production_value_ph_96_spot_True_out_of_sample",)
    train_periods = np.linspace(1,4,19)
    train_periods = np.concatenate([np.linspace(1,4,19), np.linspace(4.5,10,12)])
    forecaster_types = [f"SOTA{str(round(float(train_period),2)).replace(".","_")}year" for train_period in train_periods]
    experiments = [f"{forecaster_type}_{agent}_{scenario_name}" for agent in agents for forecaster_type in forecaster_types]

    rfp = create_rfp(scenario_name=scenario_name, layout_file=layout_file)
    results_folder = f"setup_files/results/{layout_file.split('.')[0]}"
    use_optimized_capacities = True
    if use_optimized_capacities:
        rfp.set_capacities_from_file(f"{results_folder}/optimal_capacities-chosen.csv")
    ppa_prices = pd.read_csv(f"{results_folder}/ppa_prices-risk_neutral.csv")
    for name, ppa in rfp.get_ppas().items():
        resource = ppa.parameters.get("consumes")
        if resource in ('wind', 'solar'):
            price = ppa_prices[resource].iloc[0]
        else:
            price = ppa_prices["baseload"].iloc[0]
        ppa.parameters["price"] = np.round(price, 2)

    # production_distribution = {}
    # for agent in agents:
    #     experiment_names = [f"{forecaster_type}_{agent}_100" for forecaster_type in forecaster_types]
    #     cols = [str(round(float(train_period),2)).replace(".","_") for train_period in train_periods]
    #     production_dists = pd.DataFrame(columns=cols, index=range(1,13))
    #     for exp_ix, experiment_name in enumerate(experiment_names):
    #         trajectory = load_trajectories(experiment_name)[0]
    #         horizon_days = len(trajectory.reward)
    #         T = range(1, horizon_days+1)
    #         power_import = []
    #         for t in T:
    #             power_import += list(trajectory.env_info[t]['power_consumption'])
    #         hourly_index = pd.to_datetime(pd.date_range(start=trajectory.env_info[1]['time'], periods=horizon_days*24, freq='h'))
    #         monthly_imports = pd.DataFrame(data=power_import, index=hourly_index, columns=["import"]).groupby(hourly_index.month).sum()
    #         production_dists.iloc[:, exp_ix] = monthly_imports["import"].values
    #     production_distribution[agent] = production_dists

    opt_models = []
    for exp_ix, experiment_name in enumerate(experiments):
        trajectories = load_trajectories(experiment_name)
        stats = load_stats(experiment_name, csv_version=False)

        trajectory_summaries = []
        emissions_summaries = []
        capture_price_summaries = []
        ppa_capacity_factors = []
        
        unique_years = np.unique([trajectories[0].env_info[x]['time'].year for x in range(len(trajectories[0].reward))])
        for ix, year in enumerate(unique_years):
            if year == 2020:
                print("2020")
            print(f"Evaluating year {year} of experiment {experiment_name}")
            if exp_ix == 0:
                opt_model, trajectory_summary, emissions_summary, capture_price_summary, ppa_capacity_factor = get_yearly_summary(trajectory=trajectories[0], stats=stats[0], rfp=rfp, year=year)
                opt_models.append(opt_model)
                if documentation:
                    if not os.path.exists(f"documentation/{documentation}"):
                        os.makedirs(f"documentation/{documentation}")
                    prices = [p.value for p in opt_model.inst.electricity_price.values()]
                    imports = list(opt_model.decision_results.link_production["Grid Connection Point"])
                    fig, ax = plt.subplots(figsize=(12,8))
                    plt.scatter(prices, imports, s=2)
                    plt.xlabel("Prices [€/MWh]")
                    plt.ylabel("Net Power Consumption [MW]")
                    plt.savefig(f"documentation/{documentation}/{year}_scatter_{exp_ix}.png")
                    plt.close()
                    ss = [[p,i] for p,i in zip(prices, imports)]
                    max_import = max(imports)
                    min_import = min(imports)
                    sort_ss = sorted(ss) # Sort by prices, index 0
                    data = np.asarray(sort_ss)
                    fig, ax = plt.subplots(figsize=(12,8))
                    for jx, datapoint in enumerate(data):
                        plt.bar(jx, datapoint[0] * datapoint[1]/max_import, width=0.1, color="steelblue", edgecolor="steelblue")
                        plt.bar(jx, datapoint[0], width=0.1, color="lightgrey", edgecolor="lightgrey")
                    # plt.bar(0,0, width=0.1, color="steelblue", label="Consuming")
                    # plt.bar(0,0, width=0.1, color="lightgrey", label="Not Consuming")
                    plt.plot(data[:,0], label="100% load", linestyle="-",alpha=0.5, color='black')
                    plt.plot(data[:,0] * 0.9, label="90% load", linestyle="-.",alpha=0.5, color='black')
                    plt.plot(data[:,0] * 0.5, label="50% load", linestyle=":",alpha=0.5, color='black')
                    plt.plot(data[:,0] * min_import/max_import, label="Min. load", linestyle="--",alpha=0.5, color='black')
                    plt.xlabel("Sorted hours")
                    plt.ylabel("Electricity Price [€/MWh]")
                    plt.legend(loc="upper left")
                    plt.savefig(f"documentation/{documentation}/{year}_duration_curve_{exp_ix}.png")
                    plt.close()
            else:
                _, trajectory_summary, emissions_summary, capture_price_summary, ppa_capacity_factor = get_yearly_summary(trajectory=trajectories[0], stats=stats[0], rfp=rfp, year=year, opt_model=opt_models[ix])
            trajectory_summaries.append(trajectory_summary)
            emissions_summaries.append(emissions_summary)
            capture_price_summaries.append(capture_price_summary)
            ppa_capacity_factors.append(ppa_capacity_factor)
            # Make sure the output directory exists
            os.makedirs(f"evaluation_scripts/processed_results/{experiment_name}_years", exist_ok=True)
        pd.DataFrame(trajectory_summaries).to_csv(f"evaluation_scripts/processed_results/{experiment_name}_years/trajectory_summary.csv", index=False)
        pd.DataFrame(emissions_summaries).to_csv(f"evaluation_scripts/processed_results/{experiment_name}_years/emissions_summary.csv", index=False)
        pd.DataFrame(capture_price_summaries).to_csv(f"evaluation_scripts/processed_results/{experiment_name}_years/capture_price_summary.csv", index=False)
        pd.DataFrame(ppa_capacity_factors).to_csv(f"evaluation_scripts/processed_results/{experiment_name}_years/ppa_capacity_factors.csv", index=False)
        print(f"Evaluation of {experiment_name} finished")





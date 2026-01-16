""" Necessary path addendum if we want to run this script not from the root.
    To run from root call:
    python -m test_scripts.SCRIPTNAME """
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common_scripts.utils import load_trajectories, load_stats
from common_scripts.RFP_initialization import create_rfp
from evaluation_scripts.get_experiment_summary import print_trajectory_summary
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common_scripts.utils import set_plotting_style
set_plotting_style()

if __name__ == '__main__':
    documentation = False
    single_experiment_evaluation = False
    document_optimal_strategy = False

    # scenarios = ["80percent_ammonia", "70percent_ammonia", "60percent_ammonia", "50percent_ammonia", "40percent_ammonia", "30percent_ammonia", "20percent_ammonia"]
    scenarios = ["base_case_no_spot", "no_solar_power", "high_RE_level"]

    """ Change the experiment name to the one we want to evaluate: """
    # agents = ("DeterministicHA_hourly_target_ph_72_spot_True",
    #           "DeterministicHA_production_value_ph_72_spot_True",
    #           "RecourseAgent5_DAbidding_production_value_ph_72_spot_True",
    #           "DeterministicHA_production_value_ph_72_spot_True_ispmetric_max",
    #           "StrikePriceBiddingAgent1_SP1_production_value_ph_72_spot_True_ispmetric_max",
    # )
    agents = ("DeterministicHA_production_value_ph_72_spot_False",
              "RecourseAgent1_production_value_ph_72_spot_False",
              "StrikePriceBiddingAgent1_SP1_production_value_ph_72_spot_False",
    )

    #%% Trajectory stats:
    for scenario in scenarios:
        opt_models = []
        rfp = create_rfp(scenario_name=scenario)
        experiments = [f"scenario_{scenario}_{agent}" for agent in agents]
        for exp_ix, experiment_name in enumerate(experiments):
            trajectories = load_trajectories(experiment_name)
            stats = load_stats(experiment_name, csv_version=False)

            trajectory_summaries = []
            emissions_summaries = []
            capture_price_summaries = []
            ppa_capacity_factors = []
            
            for ix in range(20):
                if exp_ix == 0:
                    opt_model, trajectory_summary, emissions_summary, capture_price_summary, ppa_capacity_factor = print_trajectory_summary(trajectory=trajectories[ix], stats=stats[ix], rfp=rfp)
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
                    _, trajectory_summary, emissions_summary, capture_price_summary, ppa_capacity_factor = print_trajectory_summary(trajectory=trajectories[ix], stats=stats[ix], rfp=rfp, opt_model=opt_models[ix])
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

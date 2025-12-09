""" Necessary path addendum if we want to run this script not from the root. 
    To run from root call:
    python -m test_scripts.SCRIPTNAME """
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme("paper", font_scale=1.5, style="darkgrid")
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

experiment_name = "test_RecourseAgent1_production_value_ph_96_spot_True"
# experiment_name = "test_RecourseAgent5_production_value_ph_96_spot_True"
# experiment_name = "testDomainsDecisionRuleAgent"
# experiment_name = "test_DeterministicHA_hourly_target_ph_96_spot_True"
# experiment_name = "test_DeterministicHA_production_value_ph_96_spot_True"
# experiment_name = "test_StochasticHA5_production_value_ph_96_spot_True"

results = pd.read_csv(f"evaluation_scripts/processed_results/{experiment_name}.csv")
# results2 = pd.read_csv(f"evaluation_scripts/processed_results/{experiment_name2}.csv")

fig, ax = plt.subplots(figsize=(16,12))
sns.histplot(results["Percentage Obtained"], label="Obtained profits", color="red", alpha=0.8, bins=12)
plt.axvline(np.mean(results["Percentage Obtained"]), label="Average obtained profit", lw=5, color='darkred')
plt.xlabel(f"% of optimal profits")
plt.ylabel("Occurences")
plt.legend()
plt.savefig(f"documentation/profit_dists/{experiment_name}_percentages.png")
plt.close()

fig, ax = plt.subplots(figsize=(16,12))
sns.histplot(results["Optimal Profit"].values/1e6, label="Optimal profits", color="blue", alpha=0.8, bins=12)
plt.axvline(np.mean(results["Optimal Profit"])/1e6, label="Average optimal profit", lw=5, color='darkblue')
plt.xlabel(f"€ (million)")
plt.ylabel("Occurences")
plt.legend()
plt.savefig(f"documentation/profit_dists/{experiment_name}_optimals.png")
plt.close()

fig, ax = plt.subplots(figsize=(16,12))
sns.histplot(results["Realized Profit"].values/1e6, label="Obtained profits", color="green", alpha=0.8, bins=12)
plt.axvline(np.mean(results["Realized Profit"].values)/1e6, label="Average profit", lw=5, color='darkgreen')
plt.axvline(np.percentile(results["Realized Profit"].values, 10)/1e6, label="P90 profit", lw=5, color='black')
plt.xlabel(f"€ (million)")
plt.ylabel("Occurences")
plt.legend()
plt.savefig(f"documentation/profit_dists/{experiment_name}_realized.png")
plt.close()

fig, ax = plt.subplots(figsize=(16,12))
sns.histplot(results["EBITDA"].values/1e6, label="EBITDA", color="green", alpha=0.8, bins=12)
plt.axvline(np.mean(results["EBITDA"].values)/1e6, label="Average EBITDA", lw=5, color='darkgreen')
plt.axvline(np.percentile(results["Realized Profit"].values, 10)/1e6, label="P90 EBITDA", lw=5, color='black')
plt.xlabel(f"€ (million)")
plt.ylabel("Occurences")
plt.legend()
plt.savefig(f"documentation/profit_dists/{experiment_name}_EBITDA.png")
plt.close()

fig, ax = plt.subplots(figsize=(16,12))
sns.histplot(results["Optimal Profit"].values/1e6, label="Optimal profits", color="blue", alpha=0.8, bins=12)
plt.axvline(np.mean(results["Optimal Profit"].values)/1e6, label="Average optimal profit", lw=5, color='darkblue')
sns.histplot(results["Realized Profit"].values/1e6, label="Obtained profits", color="green", alpha=0.8, bins=12)
plt.axvline(np.mean(results["Realized Profit"].values)/1e6, label="Average obtained profit", lw=5, color='darkgreen')
plt.xlabel(f"€ (million)")
plt.ylabel("Occurences")
plt.legend()
plt.savefig(f"documentation/profit_dists/{experiment_name}_real_and_opt.png")
plt.close()

fig, ax = plt.subplots(figsize=(16,12))
plt.scatter(results["Optimal Profit"].values/1e6, results["Realized Profit"].values/1e6, label="Scenarios", color="green", alpha=0.8, s=50)
plt.xlabel("Optimal Profits [€ (million)]")
plt.ylabel("Realized Profits [€ (million)]")
plt.legend()
plt.savefig(f"documentation/profit_dists/{experiment_name}_real_and_opt_scatter.png")
plt.close()

# fig, ax = plt.subplots(figsize=(16,12))
# plt.scatter(results["Optimal Profit"].values/1e6, results["Realized Profit"].values/1e6, label="Deterministic Agent", color="green", alpha=0.8, s=30)
# plt.scatter(results2["Optimal Profit"].values/1e6, results2["Realized Profit"].values/1e6, label="Stochastic Agent", color="blue", alpha=0.8, s=30)
# plt.xlabel("Optimal Profits [€ (million)]")
# plt.ylabel("Realized Profits [€ (million)]")
# plt.legend()
# plt.savefig(f"documentation/profit_dists/recourse_agents_real_and_opt_scatter.png")
# plt.close()

fig, ax = plt.subplots(figsize=(16,12))
sns.histplot(results["Short Exposure"].values, label="Short Exposure (Det. Agent)", color="green", alpha=0.5, bins=12)
# sns.histplot(results["Short Exposure"].values, label="Short Exposure (Stoch. Agent)", color="blue", alpha=0.5, bins=12)
sns.histplot(results["Long Exposure"].values, label="Long Exposure (Det. Agent)", color="red", alpha=0.5, bins=12)
# sns.histplot(results["Long Exposure"].values, label="Long Exposure (Stoch. Agent)", color="blue", alpha=0.5, bins=12)
sns.histplot(results["Balancing Exposure"].values, label="Balancing Exposure (Det. Agent)", color="blue", alpha=0.5, bins=12)
# sns.histplot(results["Balancing Exposure"].values, label="Balancing Exposure (Stoch. Agent)", color="blue", alpha=0.5, bins=12)
plt.xlabel("Exposure")
plt.ylabel("Occurences")
plt.legend()
plt.savefig(f"documentation/profit_dists/{experiment_name}_exposures.png")
plt.close()
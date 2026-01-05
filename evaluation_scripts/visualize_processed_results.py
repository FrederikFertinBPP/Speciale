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

# experiment_name = "test_RecourseAgent1_production_value_ph_96_spot_True"
# experiment_name = "test_RecourseAgent5_production_value_ph_96_spot_True"
# experiment_name = "testDomainsDecisionRuleAgent"
# experiment_name = "test_DeterministicHA_hourly_target_ph_96_spot_True"
# experiment_name = "test_DeterministicHA_production_value_ph_96_spot_True"
# experiment_name = "test_StochasticHA5_production_value_ph_96_spot_True"

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
colors = ['red'] + 2*['blue'] + 2*['green'] + 6*['orange']  # Customize as needed
documentation_dir = "profit_dists"
# experiments = ("planningsensitivity_DeterministicHA_production_value_ph_24_spot_True", 
#                 "planningsensitivity_DeterministicHA_production_value_ph_48_spot_True", 
#                 "planningsensitivity_DeterministicHA_production_value_ph_72_spot_True",
#                 "planningsensitivity_DeterministicHA_production_value_ph_96_spot_True", 
#                 )
# colors = 4*['blue']  # Customize as needed
# documentation_dir = "planning_sensitivity"

exp_ebitda = {}
VaR_90_ebitda = {}
short_exposure = {}
long_exposure = {}
cost_exposure = {}
revenue_exposure = {}
balancing_exposure = {}
exp_percentages = {}
VaR_90_percentages = {}

results_dict = {}



for experiment_name in experiments:
    f = experiment_name.split("_")
    if documentation_dir == "profit_dists":
        name = "".join(f[1:3])
    else:
        name = "".join([f[1], f[5]])
    scenario_dir = os.path.join("documentation", documentation_dir, experiment_name)
    os.makedirs(scenario_dir, exist_ok=True)
    results = pd.read_csv(f"evaluation_scripts/processed_results/{experiment_name}/trajectory_summary.csv")
    results_dict[name] = results
    # results2 = pd.read_csv(f"evaluation_scripts/processed_results/{experiment_name2}.csv")

    fig, ax = plt.subplots(figsize=(16,12))
    sns.histplot(results["Profit Percentage [%]"], label="Obtained profits", color="red", alpha=0.8, bins=12)
    plt.axvline(np.mean(results["Profit Percentage [%]"]), label="Average obtained profit", lw=5, color='darkred')
    plt.xlabel(f"% of optimal profits")
    plt.ylabel("Occurences")
    plt.legend()
    plt.savefig(os.path.join(scenario_dir, "percentages.png"))
    plt.close()
    exp_percentages[name] = np.mean(results["Profit Percentage [%]"])
    VaR_90_percentages[name] = np.percentile(results["Profit Percentage [%]"],10)

    fig, ax = plt.subplots(figsize=(16,12))
    sns.histplot(results["EBITDA [€]"].values/1e6, label="EBITDA", color="green", alpha=0.8, bins=12)
    plt.axvline(np.mean(results["EBITDA [€]"].values)/1e6, label="Average EBITDA", lw=5, color='darkgreen')
    plt.axvline(np.percentile(results["EBITDA [€]"].values, 10)/1e6, label="P90 EBITDA", lw=5, color='black')
    plt.xlabel(f"€ (million)")
    plt.ylabel("Occurences")
    plt.legend()
    plt.savefig(os.path.join(scenario_dir, "EBITDA.png"))
    plt.close()
    exp_ebitda[name] = np.mean(results["EBITDA [€]"].values)/1e6
    VaR_90_ebitda[name] = np.percentile(results["EBITDA [€]"].values, 10)/1e6

    fig, ax = plt.subplots(figsize=(16,12))
    sns.histplot(results["Optimal Profit [€]"].values/1e6, label="Optimal profits", color="blue", alpha=0.8, bins=12)
    plt.axvline(np.mean(results["Optimal Profit [€]"].values)/1e6, label="Average optimal profit", lw=5, color='darkblue')
    sns.histplot(results["EBITDA [€]"].values/1e6, label="Obtained profits", color="green", alpha=0.8, bins=12)
    plt.axvline(np.mean(results["EBITDA [€]"].values)/1e6, label="Average obtained profit", lw=5, color='darkgreen')
    plt.xlabel(f"€ (million)")
    plt.ylabel("Occurences")
    plt.legend()
    plt.savefig(os.path.join(scenario_dir, "real_and_opt.png"))
    plt.close()

    fig, ax = plt.subplots(figsize=(16,12))
    plt.scatter(results["Optimal Profit [€]"].values/1e6, results["EBITDA [€]"].values/1e6, label="Scenarios", color="green", alpha=0.8, s=50)
    plt.xlabel("Optimal Profits [€ (million)]")
    plt.ylabel("Realized Profits [€ (million)]")
    plt.legend()
    plt.savefig(os.path.join(scenario_dir, "real_and_opt_scatter.png"))
    plt.close()

    fig, ax = plt.subplots(figsize=(16,12))
    sns.histplot(results["Optimal Profit [€]"].values/1e6, label="Optimal profits", color="blue", alpha=0.8, bins=12)
    plt.xlabel("€ (million)")
    plt.ylabel("Occurences")
    plt.legend()
    plt.savefig(os.path.join(scenario_dir, "optimals.png"))
    plt.close()

    fig, ax = plt.subplots(figsize=(16,12))
    sns.histplot(results["Short Exposure [%]"].values, label="Short Exposure", color="green", alpha=0.5, bins=12)
    # sns.histplot(results["Short Exposure"].values, label="Short Exposure (Stoch. Agent)", color="blue", alpha=0.5, bins=12)
    sns.histplot(results["Long Exposure [%]"].values, label="Long Exposure", color="red", alpha=0.5, bins=12)
    # sns.histplot(results["Long Exposure"].values, label="Long Exposure (Stoch. Agent)", color="blue", alpha=0.5, bins=12)
    # if np.max(results["Balancing Exposure [%]"].values) > 0:
    #     sns.histplot(results["Balancing Exposure [%]"].values, label="Balancing Exposure", color="blue", alpha=0.5, bins=12)
    # sns.histplot(results["Balancing Exposure"].values, label="Balancing Exposure (Stoch. Agent)", color="blue", alpha=0.5, bins=12)
    plt.xlabel("Exposure")
    plt.ylabel("Occurences")
    plt.legend()
    plt.savefig(os.path.join(scenario_dir, "exposure.png"))
    plt.close()
    short_exposure[name] = np.mean(results["Short Exposure [%]"].values)
    long_exposure[name] = np.mean(results["Long Exposure [%]"].values)
    balancing_exposure[name] = np.mean(results["Balancing Exposure [%]"].values)

    fig, ax = plt.subplots(figsize=(16,12))
    sns.histplot(results["Cost Exposure [%]"].values, label="Cost Exposure", color="green", alpha=0.5, bins=12)
    sns.histplot(results["Revenue Exposure [%]"].values, label="Revenue Exposure", color="red", alpha=0.5, bins=12)

    plt.xlabel("Exposure")
    plt.ylabel("Occurences")
    plt.legend()
    plt.savefig(os.path.join(scenario_dir, "exposure_CR.png"))
    plt.close()
    cost_exposure[name] = np.mean(results["Cost Exposure [%]"].values)
    revenue_exposure[name] = np.mean(results["Revenue Exposure [%]"].values)



fig, ax = plt.subplots(figsize=(16,12))
sns.histplot(results["Optimal Profit [€]"].values/1e6, label="Optimal EBITDA", color="blue", alpha=0.8, bins=12)
plt.axvline(np.mean(results["Optimal Profit [€]"])/1e6, label="Average optimal EBITDA", lw=5, color='darkblue')
plt.xlabel(f"€ (million)")
plt.ylabel("Occurences")
plt.legend()
plt.savefig(f"documentation/{documentation_dir}/optimals.png")
plt.close()

exp_ebitda["Optimal"] = np.mean(results["Optimal Profit [€]"])/1e6
VaR_90_ebitda["Optimal"] = np.percentile(results["Optimal Profit [€]"],10)/1e6


fig, ax = plt.subplots(figsize=(12,8))
x = np.arange(len(exp_ebitda))
width = 0.35
ax.bar(x - width/2, exp_ebitda.values(), width, label="Expected EBITDA")
ax.bar(x + width/2, VaR_90_ebitda.values(), width, label="VaR EBITDA (90%)")
ax.set_xticks(x)
ax.set_xticklabels(exp_ebitda.keys(), rotation=90)
ax.set_yticklabels(ax.get_yticklabels(),rotation=90)
plt.legend()
plt.ylabel("€ (million)")
plt.tight_layout()
plt.savefig(f"documentation/{documentation_dir}/EBITDA_hist_plot.png")
plt.close()

fig, ax = plt.subplots(figsize=(12,8))
# plt.title("Percentage of traded power in Intrada")
x = np.arange(len(balancing_exposure))
ax.bar(x, np.asarray(list(balancing_exposure.values()))*100, label=r"Intraday Exposure")
ax.set_xticks(x)
ax.set_xticklabels(balancing_exposure.keys(), rotation=90)
ax.set_yticklabels(ax.get_yticklabels(),rotation=90)
plt.legend()
plt.ylabel("Exposure [%]")
plt.tight_layout()
plt.savefig(f"documentation/{documentation_dir}/balancing_exposure.png")
plt.close()

fig, ax = plt.subplots(figsize=(12,8))
x = np.arange(len(long_exposure))
width = 0.35
ax.bar(x - width/2, np.asarray(list(long_exposure.values()))*100, width, label="Long Exposure")
ax.bar(x + width/2, np.asarray(list(short_exposure.values()))*100, width, label="Short Exposure")
ax.set_xticks(x)
ax.set_xticklabels(long_exposure.keys(), rotation=90)
ax.set_yticklabels(ax.get_yticklabels(),rotation=90)
plt.legend()
plt.ylabel("Exposure [%]")
plt.tight_layout()
plt.savefig(f"documentation/{documentation_dir}/exposures.png")
plt.close()

df = pd.DataFrame()
for name, r in results_dict.items():
    df[name] = r["EBITDA [€]"] / 1e6
df["Optimal"] = results["Optimal Profit [€]"]/1e6

fig, ax = plt.subplots(figsize=(12,8))
# Create boxplot with patch_artist=True to allow coloring
box = ax.boxplot(df.values, patch_artist=True)
x = np.arange(len(df.columns)) + 1
# Apply colors to each box
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
# Style median lines
for median in box['medians']:
    median.set_color('black')
    median.set_linewidth(2)
ax.set_xticks(x)
ax.set_xticklabels(df.columns, rotation=90)
ax.set_yticklabels(ax.get_yticklabels(), rotation=90)  # Rotate y-axis labels
plt.ylabel("€ (million)")  # Set the y-axis label
plt.tight_layout()  # Adjust layout
plt.savefig(f"documentation/{documentation_dir}/ebitda_boxplot.png")  # Save the figure
plt.close()  # Close the plot


df = pd.DataFrame()
for name, r in results_dict.items():
    df[name] = r["Scope 2 Emissions [tCO2]"]

fig, ax = plt.subplots(figsize=(12,8))
box = ax.boxplot(df.values, patch_artist=True)
x = np.arange(len(df.columns)) + 1
# Apply colors to each box
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
# Style median lines
for median in box['medians']:
    median.set_color('black')
    median.set_linewidth(2)
ax.set_xticks(x)
ax.set_xticklabels(df.columns, rotation=90)
plt.ylabel("tCO2")
yticks=(np.round(np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 4)/1e4,0)*1e4).astype(int)
plt.yticks(ticks=yticks, labels=yticks, rotation=90)
plt.tight_layout()
plt.savefig(f"documentation/{documentation_dir}/emissions_boxplot.png")
plt.close()

df = pd.DataFrame()
for name, r in results_dict.items():
    df[name] = r["Profit Percentage [%]"]

fig, ax = plt.subplots(figsize=(12,8))
box = ax.boxplot(df.values, patch_artist=True)
x = np.arange(len(df.columns)) + 1
# Apply colors to each box
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
# Style median lines
for median in box['medians']:
    median.set_color('black')
    median.set_linewidth(2)
ax.set_xticks(x)
ax.set_xticklabels(df.columns, rotation=90)
ax.set_yticklabels(ax.get_yticklabels(),rotation=90)
plt.ylabel("%")
ax.set_ylim(top=100)
plt.tight_layout()
plt.savefig(f"documentation/{documentation_dir}/percentage_boxplots.png")
plt.close()

df = pd.DataFrame()
for name, r in results_dict.items():
    df[name + "_emissions"] = r["Scope 2 Emissions [tCO2]"]
    df[name + "_shortexposure"] = r["Short Exposure [%]"]
    df[name + "_longexposure"] = r["Long Exposure [%]"]
    df[name + "_ebitda"] = r["EBITDA [€]"] / 1e6

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(df["RecourseAgent5DAbidding_ebitda"], df["RecourseAgent5DAbidding_emissions"], color='blue', alpha=0.7, s=30)
plt.ylabel("tCO2 Emissions")
plt.xlabel("EBITDA (€ million)")
plt.title("Emissions vs EBITDA for DA Bidding Recourse Agent")
plt.savefig(f"documentation/{documentation_dir}/emissions_vs_ebitda_DAbiddingagent.png")
plt.show()

# Short Exposure for various agents vs EBITDA
fig, ax = plt.subplots(figsize=(12,8))

x, y = df["DeterministicHAproduction_ebitda"], df["DeterministicHAproduction_shortexposure"]
ax.scatter(x,y, color='blue', alpha=0.7, s=10, label="Deterministic HA")
ax.scatter(np.mean(x), np.mean(y), color='blue', marker='x', s=200, lw=3)
x, y = df["StochasticHA5production_ebitda"], df["StochasticHA5production_shortexposure"]
ax.scatter(x, y, color='green', alpha=0.7, s=10, label="Stochastic HA5")
ax.scatter(np.mean(x), np.mean(y), color='green', marker='x', s=200, lw=3)
x, y = df["RecourseAgent1production_ebitda"], df["RecourseAgent1production_shortexposure"]
ax.scatter(x, y, color='red', alpha=0.7, s=10, label="Recourse Agent")
ax.scatter(np.mean(x), np.mean(y), color='red', marker='x', s=200, lw=3)
x, y = df["RecourseAgent5DAbidding_ebitda"], df["RecourseAgent5DAbidding_shortexposure"]
ax.scatter(x, y, color='orange', alpha=0.7, s=10, label="Recourse Agent DA bidding")
ax.scatter(np.mean(x), np.mean(y), color='orange', marker='x', s=200, lw=3)
x, y = df["RecourseAgent5production_ebitda"], df["RecourseAgent5production_shortexposure"]
ax.scatter(x, y, color='brown', alpha=0.7, s=10, label="Recourse Agent Stoch.")
ax.scatter(np.mean(x), np.mean(y), color='brown', marker='x', s=200, lw=3)

plt.xlabel("EBITDA (€ million)")
plt.ylabel("Short Exposure [%]")
plt.title("EBITDA vs Exposure")
plt.legend()
plt.savefig(f"documentation/{documentation_dir}/ebitda_vs_shortexp.png")
plt.close()


# Long Exposure for various agents vs EBITDA
fig, ax = plt.subplots(figsize=(12,8))

x, y = df["DeterministicHAproduction_ebitda"], df["DeterministicHAproduction_longexposure"]
ax.scatter(x,y, color='blue', alpha=0.7, s=10, label="Deterministic HA")
ax.scatter(np.mean(x), np.mean(y), color='blue', marker='x', s=200, lw=3)
x, y = df["StochasticHA5production_ebitda"], df["StochasticHA5production_longexposure"]
ax.scatter(x, y, color='green', alpha=0.7, s=10, label="Stochastic HA5")
ax.scatter(np.mean(x), np.mean(y), color='green', marker='x', s=200, lw=3)
x, y = df["RecourseAgent1production_ebitda"], df["RecourseAgent1production_longexposure"]
ax.scatter(x, y, color='red', alpha=0.7, s=10, label="Recourse Agent")
ax.scatter(np.mean(x), np.mean(y), color='red', marker='x', s=200, lw=3)
x, y = df["RecourseAgent5DAbidding_ebitda"], df["RecourseAgent5DAbidding_longexposure"]
ax.scatter(x, y, color='orange', alpha=0.7, s=10, label="Recourse Agent DA bidding")
ax.scatter(np.mean(x), np.mean(y), color='orange', marker='x', s=200, lw=3)
x, y = df["RecourseAgent5production_ebitda"], df["RecourseAgent5production_longexposure"]
ax.scatter(x, y, color='brown', alpha=0.7, s=10, label="Recourse Agent Stoch.")
ax.scatter(np.mean(x), np.mean(y), color='brown', marker='x', s=200, lw=3)

plt.xlabel("EBITDA (€ million)")
plt.ylabel("Long Exposure [%]")
plt.title("EBITDA vs Exposure")
plt.legend()
plt.savefig(f"documentation/{documentation_dir}/ebitda_vs_longexp.png")
plt.close()

print("Done")
""" Necessary path addendum if we want to run this script not from the root. 
    To run from root call:
    python -m test_scripts.SCRIPTNAME """

from logging import info
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

LONGTERM_SIMULATION_EVERY_N_DAYS = 7  # how often to run the long-term simulation (in days)
SIMULATION_HORIZON = 13*31*24

filename = "historical_data/clean_dataframes/backcasting_timeseries.csv"
real_data = pd.read_csv(filename, index_col=0)
real_data.index = pd.to_datetime(real_data.index)

time = pd.Timestamp('2017-01-01 00:00:00')
forecaster_types = ("forecaster", "persistence", "prophet")
targets = ["price", "wind", "solar"]

mae = dict(zip(forecaster_types, [[], [], []]))
mape = dict(zip(forecaster_types, [[], [], []]))
mse = dict(zip(forecaster_types, [[], [], []]))
rmse = dict(zip(forecaster_types, [[], [], []]))
error_metrics = {"MAE": mae, "MAPE": mape, "MSE": mse, "RMSE": rmse}


quant_info = dict(zip(forecaster_types, [[], [], []]))
real_quants = []

df_online = real_data.loc[time:]
online_days = df_online.resample("D").first().index  # unique days in online period
n_days = len(online_days)

print(f"Starting online learning loop over {n_days} days...\n")

for day_idx, current_day in enumerate(online_days):
    day_str = current_day.strftime("%Y-%m-%d")
    for fc_idx, forecaster_type in enumerate(forecaster_types):
        forecast_path = f"scenario_data/Historicals/{forecaster_type}/"
        if day_idx % LONGTERM_SIMULATION_EVERY_N_DAYS == 0:
            year_simulation = pd.read_csv(forecast_path + f"long-term-sim_{day_str}.csv", index_col=0)
            year_simulation.index = pd.to_datetime(year_simulation.index)
            year_simulation = year_simulation[targets]
            sim_quantiles = year_simulation.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
            if fc_idx == 0:
                print(f"Time: {current_day}")
                actuals_year = real_data.loc[year_simulation.index]
                actuals_year = actuals_year[targets]
                actual_quantiles = actuals_year.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
                real_quants.append(actual_quantiles)
            quant_info[forecaster_type].append(sim_quantiles)
        
        forecast = pd.read_csv(f"{forecast_path}forecast_{day_str}.csv", index_col=0)
        forecast.index = pd.to_datetime(forecast.index)
        actuals = real_data.loc[forecast.index]
        forecast = forecast[targets]
        actuals = actuals[targets]

        errors = np.abs(forecast - actuals)
        mae[forecaster_type].append(errors.mean())
        mape[forecaster_type].append((errors / np.abs(actuals)).replace([np.inf, -np.inf], np.nan).mean()*100)
        mse[forecaster_type].append((errors ** 2).mean())
        rmse[forecaster_type].append(np.sqrt(mse[forecaster_type][-1]))

""" Plot results """
for target_idx, target in enumerate(targets):
    fig, axs = plt.subplots(3, 1, figsize=(15, 10), sharey=True, sharex=True)
    fig.suptitle(f"Online Forecasting Performance for {target.capitalize()}", fontsize=16)
    axs = axs.flatten()
    for fc_idx, forecaster_type in enumerate(forecaster_types):
        ax = axs[fc_idx]
        # for error_idx, (metric_name, metric_dict) in enumerate(error_metrics.items()):
        (metric_name, metric_dict) = ("RMSE", rmse)
        ddd = np.asarray(metric_dict[forecaster_type])[:, target_idx]
        ax.plot(online_days[:len(ddd)], ddd, label=f"{forecaster_type}")
        if fc_idx == len(forecaster_types) - 1:
            ax.set_xlabel("Date")
        ax.set_ylabel("Error (€ for price, CF for wind/solar)")
        ax.set_title(f"Online Forecasting Performance ({metric_name})")
        ax.legend()
        ax.grid()
    plt.savefig(f"documentation/online_forecasting_performance/rmse_{target}.png")
    plt.close()

for target_idx, target in enumerate(targets):
    fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharey=True, sharex=True)
    fig.suptitle(f"Long-term Projection Quantiles for {target.capitalize()}", fontsize=16)
    axs = axs.flatten()

    real_ = [q[target] for q in real_quants]
    quants = real_[0].index
    quant_colors = ["blue", "orange", "green", "red", "purple"]

    for fc_idx, forecaster_type in enumerate(forecaster_types):
        ax = axs[fc_idx]

        ddd = [q[target] for q in quant_info[forecaster_type]]
        for q_idx, q in enumerate(quants):
            quant_values = [qq.loc[q] for qq in ddd]
            quant_values_real = [qq.loc[q] for qq in real_]
            ax = axs[fc_idx]
            ax.plot(online_days[::LONGTERM_SIMULATION_EVERY_N_DAYS][:len(quant_values)], quant_values, label=f"{forecaster_type} - {q}", color=quant_colors[q_idx])
            ax.plot(online_days[::LONGTERM_SIMULATION_EVERY_N_DAYS][:len(quant_values_real)], quant_values_real, label=f"Actual - {q}", linestyle="--", color=quant_colors[q_idx])
        if fc_idx == len(forecaster_types) - 1:
            ax.set_xlabel("Date")
        ax.set_ylabel("Prediction Value (€ for price, CF for wind/solar)")
        ax.set_title(f"Long-term Projection Quantiles {forecaster_type}")
        ax.plot(df_online.index[:24*LONGTERM_SIMULATION_EVERY_N_DAYS*len(quant_values)],
                df_online[target].rolling(window=168,min_periods=1).mean().iloc[:24*LONGTERM_SIMULATION_EVERY_N_DAYS*len(quant_values)],
                label=f"Actual (MA 168 hours)", color='black', lw=3, alpha=0.5)
        ax.legend(loc="upper left")
        ax.grid()
        ax.set_ylim(-50,500) if target == "price" else ax.set_ylim(0,1)
    plt.savefig(f"documentation/online_forecasting_performance/longterm_quantiles_{target}.png")
    plt.close()





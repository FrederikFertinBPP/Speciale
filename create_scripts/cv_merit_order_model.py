#%% Initilization and imports
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from time import time

from data_scripts.data_loader import HistoricalData

from sklearn.model_selection import cross_validate
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Ridge


#%% Data retrieval - There is only data from 2015 and forward for Portugal
t_s = time()
start   = pd.Timestamp('20150101', tz='UTC')
end     = pd.Timestamp('20251231', tz='UTC')
data_object = HistoricalData(start=start, end=end, country_code='PT', server='ENTSOE')
print(f"Data retrieval and preprocessing took {time()-t_s:.2f} seconds.")

df_features = data_object.data.loc[data_object.data.index.year <= 2024, ["solar", "wind", "Actual Load", "Residual Load", "gas_with_ets"]]
X = df_features
y = data_object.data.loc[data_object.data.index.year <= 2024, ["price"]]

#%% Cross-validation of different models and feature sets
estimators = [{"model": LinearRegression, "params": {}},
                {"model": Ridge, "params": {"alpha": 10.0}},
                {"model": Ridge, "params": {"alpha": 1000.0}},
                {"model": XGBRegressor, "params": {"tree_method": "hist", "reg_alpha": 0}},
                {"model": XGBRegressor, "params": {"tree_method": "hist", "reg_alpha": 1000.0}},
                {"model": XGBRegressor, "params": {"tree_method": "hist", "reg_alpha": 5000.0, "reg_lambda": 1.0}},
                {"model": XGBRegressor, "params": {"reg_alpha": 0, "booster": "gblinear"}}, # Best XGB so far - slightly worse than Ridge
                {"model": XGBRegressor, "params": {"reg_alpha": 1.0, "booster": "gblinear"}},
                {"model": XGBRegressor, "params": {"reg_alpha": 10.0, "booster": "gblinear"}},
                {"model": XGBRegressor, "params": {"reg_alpha": 0.1, "booster": "gblinear", "reg_lambda": 0.1}},
                {"model": XGBRegressor, "params": {"reg_alpha": 0.1, "booster": "gblinear", "reg_lambda": 1.0}},
                {"model": XGBRegressor, "params": {"reg_alpha": 0.1, "booster": "gblinear", "reg_lambda": 10.0}},
]
feature_sets = [["solar", "wind", "Actual Load", "Residual Load", "gas_with_ets"],
                ["solar", "wind", "Actual Load", "gas_with_ets"],
                ["solar", "wind", "Actual Load", "Residual Load"],
                ["solar", "wind", "Actual Load"],
                ["solar", "wind", "gas_with_ets"],
                ["solar", "wind"],
                ["Residual Load", "gas_with_ets"],
]
feature_estimator = {str(feats): {"estimator": None, "cv_score": float("inf"), "train_score": float("inf")} for feats in feature_sets}
estimator_features = {est["model"]().__class__.__name__ + str(est["params"]): {"features": None, "cv_score": float("inf"), "train_score": float("inf")} for est in estimators}
res = {}
best = {"estimator": None, "features": None, "cv_score": float("inf"), "train_score": float("inf")}
print("\n\nStarting model evaluation with cross-validation...\n")
for feat_set in feature_sets:
    X_subset = X[feat_set]
    for est in estimators:
        estimator = est["model"](**est["params"])
        name = estimator.__class__.__name__ + str(est["params"])
        cv_res = cross_validate(estimator, X=X_subset, y=y, cv=20, scoring="neg_mean_absolute_error", return_train_score=True)
        res[name] = cv_res
        score = -cv_res["test_score"].mean()
        train_score = -cv_res["train_score"].mean()
        if score < best["cv_score"]:
            best["estimator"] = name
            best["features"] = feat_set
            best["cv_score"] = score
            best["train_score"] = train_score
        if score < feature_estimator[str(feat_set)]["cv_score"]:
            feature_estimator[str(feat_set)]["estimator"] = name
            feature_estimator[str(feat_set)]["cv_score"] = score
            feature_estimator[str(feat_set)]["train_score"] = train_score
        if score < estimator_features[name]["cv_score"]:
            estimator_features[name]["features"] = feat_set
            estimator_features[name]["cv_score"] = score
            estimator_features[name]["train_score"] = train_score
        print(name + " with features " + str(feat_set), 
            "\nTrain MAE:\t", -cv_res["train_score"].mean(), 
            "\nCV MAE:\t", score, "\n")

#%% Residual analysis of best estimator

# Make a prediction on the full training set with the best mode and features and plot the QQ plot
best_estimator_class = best["estimator"].split("{")[0]
best_estimator_params = eval("{" + best["estimator"].split("{")[1])
best_estimator = eval(best_estimator_class)(**best_estimator_params)
# best_estimator = LinearRegression() # Almost as good, much simpler.
best_estimator.fit(X[best["features"]], y)
y_pred = best_estimator.predict(X[best["features"]])

from common_scripts.utils import set_plotting_style
set_plotting_style()
from scipy import stats
import pmdarima as pm
import pandas as pd
import matplotlib.pyplot as plt

residuals = y.values.flatten() - y_pred.flatten()

# Standardise residuals (important: compare shape, not scale)
std_resid = (residuals - residuals.mean()) / residuals.std()

# Compute theoretical and sample quantiles manually for more control
(osm, osr), (slope, intercept, r) = stats.probplot(std_resid, dist="norm")
# Reference line through the 25th and 75th percentiles (more robust than least-squares)
q25, q75 = np.percentile(std_resid, [25, 75])
qn25, qn75 = stats.norm.ppf([0.25, 0.75])
slope_ref = (q75 - q25) / (qn75 - qn25)
intercept_ref = q25 - slope_ref * qn25
x_line = np.array([osm.min(), osm.max()])

ts = residuals
fig, axes = plt.subplots(2, 2, figsize=(15,8))
axes = axes.flatten()

ax = axes[0]
ax.scatter(y.index, y.values.flatten() - y_pred.flatten(), s=1)
ax.set_ylabel("€/MWh")
ax.set_xlabel("Date")
ax.set_title("Residuals")

fig.tight_layout(pad=4.0, rect=[0.03, 0.03, 0.97, 0.95])
pm.plot_acf(ts, ax=axes[2], lags=48, show=False)
pm.plot_pacf(ts, ax=axes[3], lags=48, show=False)
axes[2].set_xlabel("Lag")
axes[3].set_xlabel("Lag")

ax = axes[1]
ax.scatter(osm, osr, s=2, alpha=0.4, color="steelblue", label="Residuals")
ax.plot(x_line, slope_ref * x_line + intercept_ref, 'r-', lw=1.5, label="Reference line")

ax.set_xlabel("Theoretical quantiles")
ax.set_ylabel("Sample quantiles")
ax.set_title("Normal QQ Plot of Residuals")
ax.legend()

plt.tight_layout()
plt.savefig(f'documentation/correlation_mom_residuals.png')
plt.close()

#%% Cross-validation of OLS ensemble models by hour of day and for every year.
import copy
t_s=time()
best_xgb = {"model": XGBRegressor, "params": {"reg_alpha": 0.1, "booster": "gblinear", "reg_lambda": 0.1}}
estimator = best_xgb["model"](**best_xgb["params"])
estimator = LinearRegression()
estimators = []
cv_error_scores = {hour: None for hour in range(24)}
train_error_scores = {hour: None for hour in range(24)}
for hour in range(24):
    X_subset = X[X.index.hour == hour]
    y_subset = y[y.index.hour == hour]
    cv_res = cross_validate(estimator, X=X_subset, y=y_subset, cv=20, scoring="neg_mean_absolute_error", return_train_score=True)
    score = -cv_res["test_score"].mean()
    train_score = -cv_res["train_score"].mean()
    cv_error_scores[hour] = score
    train_error_scores[hour] = train_score
    estimators.append(copy.deepcopy(estimator))
print("CV error scores by hour:", cv_error_scores)
print("Average CV error:", np.mean(list(cv_error_scores.values())))
print("Time for CV of 24 hourly models:", time()-t_s)

t_s=time()
best_xgb = {"model": XGBRegressor, "params": {"reg_alpha": 0.1, "booster": "gblinear", "reg_lambda": 0.1}}
estimator = best_xgb["model"](**best_xgb["params"])
# estimator = LinearRegression()
estimators = []
years = y.index.year.unique()
cv_error_scores = {year: None for year in years}
train_error_scores = {year: None for year in years}
for year in years:
    X_subset = X[X.index.year == year]
    y_subset = y[y.index.year == year]
    cv_res = cross_validate(estimator, X=X_subset, y=y_subset, cv=20, scoring="neg_mean_absolute_error", return_train_score=True)
    score = -cv_res["test_score"].mean()
    train_score = -cv_res["train_score"].mean()
    cv_error_scores[year] = score
    train_error_scores[year] = train_score
    estimators.append(copy.deepcopy(estimator))
print("CV error scores by year:", cv_error_scores)
print("Average CV error:", np.mean(list(cv_error_scores.values())))
print("Time for CV of 10 yearly models:", time()-t_s)


#%% Additional residual analysis
estimator = LinearRegression()
estimator.fit(X,y)
y_pred=estimator.predict(X)
residuals = y.values.flatten() - y_pred.flatten()

residuals = pd.DataFrame(residuals, index=y.index)
means = residuals.groupby(residuals.index.month).mean()
stds = residuals.groupby(residuals.index.month).std()
import seaborn as sns
plt.figure(figsize=(10,6))
sns.lineplot(x=means.index, y=means.values.flatten(), label="Mean Residual")
sns.lineplot(x=stds.index, y=stds.values.flatten(), label="Std Dev of Residuals")
plt.xlabel("Hour of Day")
plt.ylabel("Residuals (€/MWh)")
plt.show()
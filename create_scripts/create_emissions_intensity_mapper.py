

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    from common_scripts.utils import cache_write
    from data_scripts.data_loader import HistoricalData, get_fossil_prices
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.model_selection import cross_validate
    from sklearn.feature_selection import SequentialFeatureSelector
    from sklearn.metrics import root_mean_squared_error
    import matplotlib.pyplot as plt
    import statsmodels.api as sm
    from common_scripts.utils import set_plotting_style
    set_plotting_style()

    import xgboost as xgb

    start   = pd.Timestamp('20240101', tz='UTC')
    end     = pd.Timestamp('20241231', tz='UTC')
    data_object = HistoricalData(start=start, end=end, country_code='PT', server='ENTSOE')

    df_ren_prices = pd.read_csv("historical_data/clean_dataframes/server-ENTSOEcountry-PT2024-01-01to2024-12-31.csv", index_col=0)
    df_ren_prices.index = pd.to_datetime(df_ren_prices.index, utc=True)

    df_emissions = pd.read_csv(f"historical_data/PT_2024_hourly_emissions.csv", index_col=0)
    df_emissions.index = pd.to_datetime(df_emissions.index, utc=True)

    df_features = data_object.data.loc[data_object.data.index.year == 2024, ["price", "solar", "wind", "Actual Load"]]
    X = df_features
    y = df_emissions[["Carbon intensity gCO₂eq/kWh (Life cycle)"]] / 1000 # tCO2eq/MWh
    model = LinearRegression()
    model.fit(X=X, y=y)
    y_pred = model.predict(X)
    print(root_mean_squared_error(y, y_pred))

    cache_path = os.getcwd() + "/models/plant_models/emission_factor.pkl"
    cache_write(model, cache_path, verbose=True)

    feats = {}
    res = {}
    for estimator in [LinearRegression(), xgb.XGBRegressor(tree_method="hist")]:
        feat_res = SequentialFeatureSelector(estimator, n_features_to_select=5).fit(X, y)
        chosen_features = X.columns[feat_res.get_support()].tolist()
        feats[estimator.__class__.__name__] = chosen_features
        print(estimator.__class__.__name__, "Selected features:", chosen_features)
        cv_res = cross_validate(estimator, X=X[chosen_features], y=y, cv=5, scoring="neg_root_mean_squared_error", return_train_score=True)
        res[estimator.__class__.__name__] = cv_res
        print(estimator.__class__.__name__, "CV RMSE:", -cv_res["test_score"].mean())

    df_ren_prices = df_ren_prices.loc[df_ren_prices.index.year==2024]
    df_emissions = df_emissions.loc[df_ren_prices.index]

    y_true = df_emissions["Carbon intensity gCO₂eq/kWh (direct)"]

    # Exponentially decreasing relation with wind
    fig, ax = plt.subplots(figsize=(16,12))
    plt.scatter(df_ren_prices['wind'].values,y_true.values, s=10)
    plt.xlabel("Wind Power (MW)")
    plt.ylabel("Emissions intensity (gCO2/kWh)")
    plt.savefig('documentation/co2_intensity_mapper/systemwind_vs_emissions.png')
    plt.close()

    fig, ax = plt.subplots(figsize=(16,12))
    plt.scatter(df_ren_prices['solar'].values,y_true.values, s=10)
    plt.xlabel("Solar Power (MW)")
    plt.ylabel("Emissions intensity (gCO2/kWh)")
    plt.savefig('documentation/co2_intensity_mapper/systemsolar_vs_emissions.png')
    plt.close()

    fig, ax = plt.subplots(figsize=(16,12))
    plt.scatter(df_ren_prices['solar'].values*df_ren_prices['wind'].values,y_true.values, s=10)
    plt.xlabel("Solar Power (MW)")
    plt.ylabel("Emissions intensity (gCO2/kWh)")
    plt.savefig('documentation/co2_intensity_mapper/solarXwind_vs_emissions.png')
    plt.close()

    # Second order relation with price
    fig, ax = plt.subplots(figsize=(16,12))
    plt.scatter(df_ren_prices['price'].values,y_true.values, s=10)
    plt.xlabel("Electricity Price (€/MWh)")
    plt.ylabel("Emissions intensity (gCO2/kWh)")
    plt.savefig('documentation/co2_intensity_mapper/price_vs_emissions.png')
    plt.close()

    df_ren_prices['wind_sq'] = (df_ren_prices['wind'].values * df_ren_prices['wind'].values)
    df_ren_prices['wind_exp'] = np.exp(df_ren_prices['wind'].values / np.max(df_ren_prices['wind'].values))
    df_ren_prices['price_sq'] = (df_ren_prices['price'].values * df_ren_prices['price'].values)

    X_ols = sm.add_constant(X)
    model = sm.OLS(y_true, X_ols)
    results = model.fit()
    print(results.summary())

    fig, ax = plt.subplots(figsize=(16,12))
    plt.scatter(y_true.values, y_pred,label="Model Prediction", s=10)
    plt.scatter(y_true.values, y_true.values,label="True Value", color="black", s=10)
    plt.xlabel("True emissions intensity (gCO2/kWh)")
    plt.ylabel("Predicted emissions intensity (gCO2/kWh)")
    plt.legend()
    plt.savefig('documentation/co2_intensity_mapper/prediction_performance.png')
    plt.close()

    fig, ax = plt.subplots(figsize=(16,12))
    plt.scatter(range(len(y_pred)),y_true.values-y_pred,label="Model Residuals", s=10)
    plt.xlabel("Training observations")
    plt.ylabel("Residuals (gCO2/kWh)")
    plt.legend()
    plt.savefig('documentation/co2_intensity_mapper/prediction_residuals.png')
    plt.close()

    fig, ax = plt.subplots(figsize=(16,12))
    plt.scatter(y_true.index,y_true.values/3.6,label="Hourly emissions intensity", s=10, color='red', alpha=0.2)
    plt.axhline(18, label="RFNBO requirement", color='black', lw=3, linestyle="--")
    plt.axhline(np.mean(y_true.values)/3.6, label="Average emissions intensity (direct)", color='red', lw=3)
    plt.xlabel("Date")
    plt.ylabel("Residuals (gCO2/MJ)")
    plt.legend()
    plt.xlim(y_true.index[0], y_true.index[-1])
    plt.tight_layout()
    plt.savefig('documentation/co2_intensity_mapper/emissions_data.png')
    plt.close()

    print("Done")


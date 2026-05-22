def plot_acf(ts):
    import pmdarima as pm
    import pandas as pd
    import matplotlib.pyplot as plt
    
    if type(ts) == pd.DataFrame:
        ts_tag = ts.columns[0]
    elif type(ts) == pd.Series:
        ts_tag = ts.name
    else:
        raise("Could not resolve time series name", NameError)
    # Plot residuals with self and main regressors
    fig, axes = plt.subplots(1, 3, figsize=(15,8), sharey=True)
    axes = axes.flatten()
    fig.tight_layout(pad=4.0, rect=[0.03, 0.03, 0.97, 0.95])
    pm.plot_acf(ts, ax=axes[0], lags=48, show=False)
    pm.plot_pacf(ts, ax=axes[1], lags=48, show=False)
    axes[2] = pm.autocorr_plot(ts, show=False)
    plt.savefig(f'documentation/correlation_{ts_tag}.png')
    plt.close()
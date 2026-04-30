import os, glob, csv
# import pickle
# import joblib
import numpy as np
from collections import namedtuple
from collections.abc import Iterable
import pandas as pd

fields = ('time', 'state', 'action', 'reward')
Trajectory = namedtuple('Trajectory', fields + ("env_info",))

def set_plotting_style():
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

class expando(object):
    pass

def cache_write(object, file_name, verbose=True):
    import joblib
    dn = os.path.dirname(file_name)
    if not os.path.exists(dn):
        os.mkdir(dn)
    if verbose: print("Writing cache...", file_name)
    # with lzma.open(file_name, 'wb') as f:
    #     pickle.dump(object, f)
    #     # compress_pickle.dump(object, f, compression="lzma", protocol=protocol)
    joblib.dump(object, file_name, compress=('lzma', 3))  # streams, doesn't buffer all in RAM
    if verbose:
        print("Done!")

def cache_exists(file_name):
    return os.path.exists(file_name)

def cache_read(file_name):
    import joblib
    if os.path.exists(file_name):
        # with lzma.open(file_name, 'rb') as f:
        #     return pickle.load(f)
        return joblib.load(file_name)
    return None

## Helper functions for saving/loading a time series
def load_time_series(experiment_name, exclude_empty=True):
    """
    Load most recent non-empty time series (we load non-empty since lazylog creates a new dir immediately)
    """
    files = list(filter(os.path.isdir, glob.glob(experiment_name+"/*")))
    if exclude_empty:
        files = [f for f in files if os.path.exists(os.path.join(f, "log.txt")) and os.stat(os.path.join(f, "log.txt")).st_size > 0]

    if len(files) == 0:
        return [], None
    recent = sorted(files, key=lambda file: os.path.basename(file))[-1]
    stats = []
    with open(recent + '/log.txt', 'r') as f:
        csv_reader = csv.reader(f, delimiter='\t')
        for i, row in enumerate(csv_reader):
            if i == 0:
                head = row
            else:
                def tofloat(v):
                    try:
                        return float(v)
                    except Exception:
                        return v

                stats.append( {k:tofloat(v) for k, v in zip(head, row) } )
    return stats, recent

def average_trajectories(trajectories):
    if len(trajectories) == 0:
        return None
    
    t = trajectories[0]
    # t._asdict()
    # n = max( [len(t.time) for t in trajectories] )
    trajectories2 = sorted(trajectories, key=lambda t: len(t.time))
    tlong = trajectories2[-1]
    dd = dict(state=[], action=[],reward=[])
    # keys = list(dd.keys())

    for t in range(len(tlong.time)):
        for k in ['state', 'action', 'reward']:
            avg = []
            for traj in trajectories:
                z = traj.__getattribute__(k)
                if len(z) > t:
                    avg.append(z[t])
            if len(avg) > 0:
                # avg = np.stack(avg)
                avg = np.mean(avg, axis=0)
                dd[k].append(avg)

    dd = {k: np.stack(v) for k, v in dd.items()}
    tavg = Trajectory(**dd, time=tlong.time, env_info=[])
    return tavg

def _get_dir_path(experiment_name):
    # Get latest of the experiments if there are multiple of the same name:
    dir_path = ""
    folder = os.getcwd() + "/experiments/"
    experiments = os.listdir(folder)
    t = -10**6 # Very large time compared to unix time from time.time()
    for exp in experiments: # Load only the most recent experiment
        file_list = tuple(exp.split('-'))
        if len(file_list) == 5:
            name, _, runtime, _, t_stamp = file_list
        else:
            continue
        if name == experiment_name:
            if int(t_stamp) > t: # Just get the most recent experiment - should manually be changed if we want an older run.
                t = int(t_stamp)
                dir_path = folder + experiment_name + "-runtime-" + runtime + "-tstamp-" + t_stamp
    return dir_path

def load_trajectories(experiment_name, ):
    dir_path = _get_dir_path(experiment_name)
    if len(dir_path)>0:
        return cache_read(dir_path + "/trajectories.pkl") # trajectories
    else:
        return []

def load_stats(experiment_name, csv_version=False):
    dir_path = _get_dir_path(experiment_name)
    if len(dir_path)>0:
        if csv_version:
            return pd.read_csv(dir_path + "/agent_logbook.csv", index_col=0)
        else:
            return cache_read(dir_path + "/stats.pkl") # trajectories
    else:
        return []

from collections.abc import Iterable

class Flattener:
    def __init__(self, sep=';', sep_iter='?'):
        self.sep = sep
        self.sep_iter = sep_iter

    def _convert_lists_to_arrays(self, obj):
        if isinstance(obj, dict):
            return {k: self._convert_lists_to_arrays(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            # Recursively convert elements
            converted = [self._convert_lists_to_arrays(el) for el in obj]

            # Try converting to numpy array if elements are not dicts
            if all(not isinstance(el, dict) for el in converted):
                try:
                    return np.array(converted)
                except Exception:
                    return converted
            else:
                return converted
        else:
            return obj

    def flatten(self, d, parent_key=''):
        items = []

        def _flatten(obj, current_key):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    new_key = f"{current_key}{self.sep}{k}" if current_key else k
                    _flatten(v, new_key)
            elif isinstance(obj, Iterable) and not isinstance(obj, (str, bytes)):
                for ix, v in enumerate(obj):
                    new_key = f"{current_key}{self.sep_iter}{ix}" if current_key else str(ix)
                    _flatten(v, new_key)
            else:
                items.append((current_key, obj))

        _flatten(d, parent_key)
        return dict(items)

    def unflatten(self, flat_dict):
        result = {}

        for flat_key, value in flat_dict.items():
            parts = flat_key.split(self.sep)
            current = result

            for i, part in enumerate(parts):
                if self.sep_iter in part:
                    key, idx = part.split(self.sep_iter)
                    idx = int(idx)

                    if key not in current:
                        current[key] = []

                    while len(current[key]) <= idx:
                        current[key].append(None)

                    if i == len(parts) - 1:
                        current[key][idx] = value
                    else:
                        if current[key][idx] is None:
                            current[key][idx] = {}
                        current = current[key][idx]
                else:
                    if i == len(parts) - 1:
                        current[part] = value
                    else:
                        current = current.setdefault(part, {})

        return self._convert_lists_to_arrays(result)


#%% Common functions
def log_transform(y):
    # Does not transform nicely for low absolute values.
    def _log(x):
        if x > 0:
            return np.log(x)
        elif x < 0:
            return -np.log(-x)
        else:
            return 0
    if type(y) == pd.DataFrame:
        y[y.columns[0]] = [_log(x) for x in y[y.columns[0]]]
        return y
    else:
        return [_log(x) for x in y]

def delog_transform(y):
    # Does not transform nicely for low absolute values.
    def _delog(x):
        if x > 0:
            return np.exp(x)
        elif x < 0:
            return -np.exp(-x)
        else:
            return 0
    if type(y) == pd.DataFrame:
        y[y.columns[0]] = [_delog(x) for x in y[y.columns[0]]]
        return y
    else:
        return [_delog(x) for x in y]

def laplace_rnd(mu, sigma, x):
    return mu - sigma * np.sign(x) * np.log(1 - 2 * np.abs(x))

def trigo_fit(t, b1, b2, b3, b4, b5, b6, b7):
    """
    Trigonometric seasonal fitting function.
    
    Parameters:
    - beta: list or array of coefficients [b1, b2, b3, b4, b5, b6, b7]
    - t: array-like time values
    
    Returns:
    - y: numpy array of fitted values
    """
    y = (
        b1 * np.cos(2 * np.pi * t * b2 + b6) +
        b3 * np.sin(4 * np.pi * t * b4 + b7) +
        b5
    )
    return np.asarray(y, dtype=float).flatten()  # Equivalent to MATLAB's y'

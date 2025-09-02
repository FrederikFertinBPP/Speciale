import os, glob, csv
import pickle
import numpy as np
from collections import namedtuple

fields = ('time', 'state', 'action', 'reward')
Trajectory = namedtuple('Trajectory', fields + ("env_info",))

def cache_write(object, file_name, verbose=True):
    import lzma
    dn = os.path.dirname(file_name)
    if not os.path.exists(dn):
        os.mkdir(dn)
    if verbose: print("Writing cache...", file_name)
    with lzma.open(file_name, 'wb') as f:
        pickle.dump(object, f)
        # compress_pickle.dump(object, f, compression="lzma", protocol=protocol)
    if verbose:
        print("Done!")

def cache_exists(file_name):
    return os.path.exists(file_name)

def cache_read(file_name):
    import lzma
    if os.path.exists(file_name):
        with lzma.open(file_name, 'rb') as f:
            return pickle.load(f)
    else:
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


def experiment_load(experiment_name, exclude_empty=True):
    files = list(filter(os.path.isdir, glob.glob(experiment_name + "/*")))
    if exclude_empty:
        files = [f for f in files if
                 os.path.exists(os.path.join(f, "log.txt")) and os.stat(os.path.join(f, "log.txt")).st_size > 0]
    if len(files) == 0:
        return []
    values = []
    files = sorted(files, key=lambda file: os.path.basename(file))
    for recent in files:
        # recent = sorted(files, key=lambda file: os.path.basename(file))[-1]
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

                    stats.append({k: tofloat(v) for k, v in zip(head, row)})

        tpath = recent + "/trajectories.pkl"
        if cache_exists(tpath):
            trajectories = cache_read(tpath)
        else:
            trajectories = None
        values.append( (stats, trajectories, recent) )
    return values
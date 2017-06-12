from __future__ import division
import numpy as np

def bootstrap(x, rep=100):
    n = len(x)
    reps = [np.random.choice(x,n,replace=True) for i in range(rep)]
    return reps

def get_median(x, bootraps):
    median = np.median(x)
    median_bootraps = np.median(bootraps, axis=1)
    var = np.var(median_bootraps)
    return [median, var]

def get_mean(x, bootraps):
    mean = np.mean(x)
    mean_bootraps = np.mean(bootraps, axis=1)
    var = np.var(mean_bootraps)
    return [mean, var]

def estimate_p(x, bootraps, a, b):
    mean = np.mean(x)
    reps = len(bootraps)
    mean_bootraps = np.mean(bootraps, axis=1)
    count = np.sum([a < diff < b for diff in (mean_bootraps - mean)])
    p = count/reps
    return p

import numpy as np
import pickle
from scipy.stats import norm

def load(filename):
    with open(filename, 'rb') as f:
        arr = pickle.load(f)
    return arr

# returns whether a>b with confidence alpha
def z_test(a, b, alpha=0.05):
    # unbiased sample variance
    a_var = np.var(a, ddof=1)   
    b_var = np.var(b, ddof=1)

    # z value
    z = (a.mean() - b.mean() - 0)/np.sqrt(a_var + b_var)

    print(a.mean(), a_var, b.mean(), b_var, z)
    return z > norm.ppf(1-alpha)

# interprets return for each run as a bernoulli r.v.
def trial(a):
    bools = a > 5.5
    return bools.astype(float)

# test on difference between population proportions
# null hypothesis is that they are the same
def pop_test(a, b, alpha=0.05):
    p_a = np.average(trial(a))
    p_b = np.average(trial(b))
    p = (p_a*len(a) + p_b*len(b))/(len(a)+len(b)) # pooled proportion
    standard_error = np.sqrt(p*(1-p)*(1/len(a) + 1/len(b)))
    z = (p_a - p_b)/standard_error
    return z > norm.ppf(1-alpha)

f1 = './partial_neural_feature_trail_15x15/return_per_run.npy'
f2 = './partial_neural_reward_trail_15x15/return_per_run.npy'
f3 = './partial_rnd_feature_trail_15x15/return_per_run.npy'
f4 = './partial_rnd_reward_trail_15x15/return_per_run.npy'

nf = load(f1)
nr = load(f2)
rf = load(f3)
rr = load(f4)

for a in [nf, nr, rf, rr]:
    print(np.sum(trial(a)))

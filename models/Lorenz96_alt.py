# add modules folder to Python's search path
import sys
from pathlib import Path
from os.path import dirname, realpath
script_dir = Path(dirname(realpath(__file__)))
module_dir = str(script_dir.parent)
sys.path.insert(0, module_dir + '/modules')

# import remaining modules
import simulate as sm
import filter as fl
import numpy as np
import scipy
import plot as plot

"""
A 2D non-linear problem (Henon map)
"""

# creates a Model object to feed the filter / combine the models
def get_model(x0, size, prior_cov=1.0, obs_cov=0.1, shift=0.0, obs_gap=0.2):
    # set parameters
    dim = len(x0)
    F = 8.0,
    eps = 0.0
    mu, id, zero =  np.zeros(dim), np.identity(dim), np.zeros(dim)
    half_dim = int(dim/2)
    mu_h, id_h, zero_h = np.zeros(half_dim), np.identity(half_dim), np.zeros(half_dim)
    shift = shift * np.ones(dim)

    # assign an attractor point as the starting point
    def lorenz96_f(t, x):
        #print(x)
        y = np.zeros(dim)
        y[0] = (x[1] - x[dim-2]) * x[dim-1] - x[0] + F
        y[1]= (x[2] - x[dim-1]) * x[0] - x[1] + F
        for i in range(2, dim-1):
            y[i] = (x[i+1] - x[i-2]) * x[i-1] - x[i] + F
        y[-1] = (x[0] - x[dim-3]) * x[dim-2] - x[dim-1] + F
        #print(y)
        return y


    def lorenz_96(x0, obs_gap=0.2):
        return scipy.integrate.solve_ivp(lorenz96_f, [0.0, obs_gap], x0, method='RK45', t_eval=[obs_gap]).y.T[0]

    # create a deterministic Markov chain
    prior = sm.Simulation(algorithm = lambda *args: shift + np.random.multivariate_normal(x0, prior_cov*id))
    process_noise = sm.Simulation(algorithm = lambda *args: np.random.multivariate_normal(mu, eps*id))
    func_h = lambda k, x, noise: lorenz_96(x, obs_gap) + noise
    conditional_pdf_h = lambda k, x, past: scipy.stats.multivariate_normal.pdf(x, mean = func_h(k, past, zero), cov = eps*id)

    # define the observation model
    func_o = lambda k, x, noise: np.array([x[2*i] for i in range(half_dim)]) + noise
    observation_noise = sm.Simulation(algorithm = lambda *args: np.random.multivariate_normal(mu_h, obs_cov*id_h))
    conditional_pdf_o = lambda k, y, condition: scipy.stats.multivariate_normal.pdf(y, mean = func_o(0, condition, mu_h), cov = obs_cov*id_h)

    # generates a trajectory according to the dynamic model
    def gen_path(x, length):
        path = np.zeros((length, dim), dtype = np.float32)
        for i in range(length):
            path[i] = x
            x = func_h(i, x, zero)
        return path

    mc = sm.DynamicModel(size = size, prior = prior, func = func_h, sigma = eps*id, noise_sim = process_noise, conditional_pdf = conditional_pdf_h)
    om = sm.MeasurementModel(size = size, func = func_o, sigma = obs_cov*id_h, noise_sim = observation_noise, conditional_pdf = conditional_pdf_o)
    return fl.Model(dynamic_model = mc, measurement_model = om), gen_path
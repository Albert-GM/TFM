# =============================================================================
#
# =============================================================================

# adding directory to pythonpath for allow own functions
import sys, os ,re
root_project = re.findall(r'(^\S*TFM_AGM)', os.getcwd())[0]
sys.path.append(root_project)

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import ParameterSampler
from scipy.stats import uniform, expon, randint
from src.utils.help_func import construct_dataframe
from src.models.sir_model import sir_model


OD = np.load('../../data/interim/od_matrix.npy')
df_countries = pd.read_pickle(
    '../../data/interim/country_info_final.pickle')

output_mode = 0

input_dataframe = []

# parametersampler
R0 = uniform(loc=2, scale=20)
Tr = uniform(loc=2, scale=20)
omega = expon(loc=0.01, scale=0.1)
limit_deaths = randint(low=1, high=1000)
n_closed = randint(low=0, high=20)
react_time = randint(low=1, high=30)
countries = list(df_countries['country_code'].values)

# countries = ['ESP', 'FRA', 'CHN', 'USA']
# countries = ['ESP']


param_grid = {'R0' : R0,
              'Tr' : Tr,
              'omega' : omega,
              'limit_deaths' : limit_deaths,
              'n_closed' : n_closed,
              'react_time' : react_time,
              'countries' : countries }

rng = np.random.RandomState(42)
param_list = list(ParameterSampler(param_grid, n_iter=50000,
                                   random_state=rng))


for simulation in tqdm(param_list):
    input_dataframe.append(
        sir_model(
            df_countries,
            OD,
            simulation['R0'],
            simulation['Tr'],
            simulation['omega'],
            simulation['countries'],
            1,
            simulation['limit_deaths'],
            simulation['n_closed'],
            simulation['react_time']))
    
    
    
# R0 = uniform(loc=1, scale=10)
# Tr = uniform(loc=15, scale=20)
# omega = expon(scale=0.1)
# limit_deaths = randint(low=1, high=1000)
# n_closed = randint(low=0, high=20)
# react_time = randint(low=1, high=30)   
# countries = top_k_connected(df_countries, 25)

# param_grid = {'R0' : R0,
#               'Tr' : Tr,
#               'omega' : omega,
#               'limit_deaths' : limit_deaths,
#               'n_closed' : n_closed,
#               'react_time' : react_time,
#               'countries' : countries }
# param_list = list(ParameterSampler(param_grid, n_iter=50000,
#                                    random_state=rng))

# for simulation in tqdm(param_list):
#     input_dataframe.append(
#         sir_model(
#             df_countries,
#             OD,
#             simulation['R0'],
#             simulation['Tr'],
#             simulation['omega'],
#             simulation['countries'],
#             1,
#             simulation['limit_deaths'],
#             simulation['n_closed'],
#             simulation['react_time']))


df_simulation = construct_dataframe(input_dataframe, output_mode)


df_simulation.to_pickle('../../data/processed/sir_simulation_50k_rev10.pickle')
# df_simulation.to_pickle('../../data/processed/sir_simulation_1k_rev10.pickle')



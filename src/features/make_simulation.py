# =============================================================================
# Computes a batch of simulations of the SIR model modificated according
# to a parameter space. The paramater space intends to be as broad as possible
# so that it covers all the possible realistic combinations that can occur
# during a pandemic. 
# =============================================================================


# Add project directory to pythonpath to import own functions
import sys, os ,re
root_project = re.findall(r'(^\S*TFM)', os.getcwd())[0]
sys.path.append(root_project)

import pandas as pd
import csv
from tqdm import tqdm
from sklearn.model_selection import ParameterSampler
from scipy.stats import uniform, expon, randint, truncexpon

from src.features.sird_model import SIRD_model

# Read necessary data
df_countries = pd.read_pickle(
    f"{root_project}/data/interim/country_info_final.pickle")

# Parameter space to explore
# In uniform distribution, loc=min and if max value desired is X, scale=X-loc
# In randint distribution if max value desired is X, high=X+low
R0 = uniform(loc=2, scale=16) 
Tr = uniform(loc=2, scale=28)
omega = truncexpon(loc=0.01, b=1-0.01) # Exponential truncated to maximum value b
n_closed = randint(low=0, high=20)
# react_time = randint(low=1, high=31)
react_time = randint(low=1, high=21)

countries = list(df_countries['country_code'].values) # All countries in df


# Alternative paramater space to explore based on model errors
# R0 = uniform(loc=10, scale=25-10) 
# Tr = uniform(loc=10, scale=30-10)
# omega = truncexpon(loc=0.01, b=0.05-0.01) # Exponential truncated to maximum value b
# n_closed = randint(low=0, high=20+0)
# react_time = randint(low=1, high=30+1)


param_grid = {'R0' : R0,
              'Tr' : Tr,
              'omega' : omega,
              'n_closed' : n_closed,
              'react_time' : react_time,
              'countries' : countries }


n_simulations = 10000 # specify the number of simulations to make
param_list = list(ParameterSampler(param_grid, n_iter=n_simulations))

# Features to keep
dict_keys = [
    'i_country',
    'R0',
    'Tc',
    'Tr',
    'omega',
    'inf_pow_1',
    'inf_pow_2',
    'sum_gradient_inf',
    'p_inf',
    'mort_pow_1',
    'mort_pow_2',
    'mort_pow_3',
    'sum_gradient_mort',
    'n_closed',
    'react_time',
    'total_infected',
    'total_deceased',
    'total_recovered']


file_name = 'simulation_results_REV3.csv'
# Un comment when simulating model based on errors
# file_name = 'simulation_results_v2.csv' 


# If the file not exist, write the header first
if not os.path.isfile(f"{root_project}/data/processed/{file_name}"):
    with open(f"{root_project}/data/processed/{file_name}", mode='w') as f:
        writer = csv.DictWriter(f, fieldnames= dict_keys)
        writer.writeheader()

# Make the simulation of all the paramater space
for simulation in tqdm(param_list):
    sir_model = SIRD_model(
        simulation['R0'],
        simulation['Tr'],
        simulation['omega'],
        simulation['countries'],
        simulation['n_closed'],
        simulation['react_time'])
    sir_model.simulate()
    sir_model.compute_disease_features()
    data = sir_model.get_simulation_data() # Get the data in a dict
    subset_data = {column: data[column] for column in dict_keys}
    # Write in the file a row at each iteration
    with open(f"{root_project}/data/processed/{file_name}", mode='a') as f:
        writer = csv.DictWriter(f, fieldnames= dict_keys)
        writer.writerow(subset_data)    

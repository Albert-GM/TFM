# =============================================================================
# Computes a batch of simulations of the SIR model modificated according
# to a parameter space. The paramater space intends to be as broad as possible
# so that it covers all the possible realistic combinations that can occur
# during an epidemic. 
# =============================================================================


# Add project directory to pythonpath to import own functions
import sys, os ,re
root_project = re.findall(r'(^\S*TFM-master)', os.getcwd())[0]
sys.path.append(root_project)

import pandas as pd
import csv
from tqdm import tqdm
from sklearn.model_selection import ParameterSampler
from scipy.stats import uniform, expon, randint, truncexpon

from src.models.sird_model import SIRD_model

# Read necessary data
df_countries = pd.read_pickle(
    f"{root_project}/data/interim/country_info_final.pickle")

# Paramater space to explore
# In uniform distribution if max value desired is X, scale=X-loc
# In randint distribution if max value desired is X, high=X+low
R0 = uniform(loc=2, scale=18) 
Tr = uniform(loc=2, scale=18)
# omega = expon(loc=0.01, scale=0.1)
omega = truncexpon(b=1) # Exponential truncated to maximum value b
limit_deceased = randint(low=1, high=1001)
n_closed = randint(low=0, high=20)
react_time = randint(low=1, high=31)
countries = list(df_countries['country_code'].values) # All countries in df


# # Alternative paramater space to explore based on model errors
# R0 = uniform(loc=2, scale=18) 
# Tr = uniform(loc=10, scale=10)
# omega = uniform(loc=0.2, scale=0.76-0.2) #
# limit_deceased = randint(low=1, high=500)
# n_closed = randint(low=0, high=21)
# react_time = randint(low=1, high=5)
# countries = list(df_countries['country_code'].values) # All countries in df
# # countries = ['ESP', 'FRA', 'ITA'] # Specify desired countries

param_grid = {'R0' : R0,
              'Tr' : Tr,
              'omega' : omega,
              'limit_deceased' : limit_deceased,
              'n_closed' : n_closed,
              'react_time' : react_time,
              'countries' : countries }


n_simulations = 100000 # specify the number of simulations to make
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
    'mort_pow_1',
    'mort_pow_2',
    'mort_pow_3',
    'limit_deceased',
    'n_closed',
    'react_time',
    'total_infected',
    'total_deceased',
    'total_recovered']


file_name = 'simulation_results_rev14_wide.csv'

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
        simulation['limit_deceased'],
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



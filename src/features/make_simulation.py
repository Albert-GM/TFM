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

import numpy as np
import pandas as pd
import csv
from tqdm import tqdm
from sklearn.model_selection import ParameterSampler
from scipy.stats import uniform, expon, randint

from src.models.sir_model_rev1 import SIR_model

# Read necessary data
df_countries = pd.read_pickle(
    f"{root_project}/data/interim/country_info_final.pickle")

# Paramter space to explore
R0 = uniform(loc=2, scale=20)
Tr = uniform(loc=2, scale=20)
omega = expon(loc=0.01, scale=0.1)
limit_deaths = randint(low=1, high=1000)
n_closed = randint(low=0, high=20)
react_time = randint(low=1, high=30)
countries = list(df_countries['country_code'].values)

param_grid = {'R0' : R0,
              'Tr' : Tr,
              'omega' : omega,
              'limit_deaths' : limit_deaths,
              'n_closed' : n_closed,
              'react_time' : react_time,
              'countries' : countries }


n_simulations = 100000 # specify the number of simulations to make
param_list = list(ParameterSampler(param_grid, n_iter=n_simulations,
            random_state=np.random.RandomState(42)))

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
    'limit_deaths',
    'n_closed',
    'react_time',
    'total_infected',
    'total_death',
    'total_removed']



if not os.path.isfile(f"{root_project}/data/processed/simulation_results.csv"):
    with open(f"{root_project}/data/processed/simulation_results.csv", mode='w') as f:
        writer = csv.DictWriter(f, fieldnames= dict_keys)
        writer.writeheader()


for simulation in tqdm(param_list):
    sir_model = SIR_model(
        simulation['R0'],
        simulation['Tr'],
        simulation['omega'],
        simulation['countries'],
        simulation['limit_deaths'],
        simulation['n_closed'],
        simulation['react_time'])
    sir_model.simulate()
    sir_model.compute_disease_features()
    data = sir_model.get_simulation_data()
    subset_data = {column: data[column] for column in dict_keys}
    with open(f"{root_project}/data/processed/simulation_results.csv", mode='a') as f:
        writer = csv.DictWriter(f, fieldnames= dict_keys)
        writer.writerow(subset_data)    







# columns = [
#     'initial_country',
#     'idx_country',
#     'R0',
#     'Tc',
#     'Tr',
#     'omega',
#     'inf_power_1',
#     'inf_power_2',
#     'gradient_inf',
#     'mort_power_1',
#     'mort_power_2',
#     'mort_power_3',
#     'gradient_mort',
#     'limit_deaths',
#     'n_closed',
#     'react_time',
#     'total_infected',
#     'total_death',
#     'total_recovered']
        

# if not os.path.isfile(f"{root_project}/data/processed/simulation_results.csv"):
#     with open(f"{root_project}/data/processed/simulation_results.csv", mode='w') as f:
#         sim_writer = csv.writer(f, delimiter=',')
#         sim_writer.writerow(columns) 



# for simulation in tqdm(param_list):
#     sir_model = SIR_model(
#         simulation['R0'],
#         simulation['Tr'],
#         simulation['omega'],
#         simulation['countries'],
#         simulation['limit_deaths'],
#         simulation['n_closed'],
#         simulation['react_time'])
#     sir_model.simulate()
#     sir_model.compute_disease_features()
#     data = sir_model.get_simulation_data()
#     with open(f"{root_project}/data/processed/simulation_results.csv", mode='a') as f:
#         sim_writer = csv.writer(f, delimiter=',')
#         sim_writer.writerow(data)
    
 

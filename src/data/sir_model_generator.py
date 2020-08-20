#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 11:54:56 2020

@author: agm
"""

import numpy as np
import seaborn as sns
import pandas as pd
import itertools
from tqdm import tqdm
from sir_model_func import sir_model

sns.set()

OD = np.load("/Users/agm/Documents/KSchool/TFM/data_out/od_matrix.npy")
df = pd.read_pickle(
    "/Users/agm/Documents/KSchool/TFM/data_out/fulldataset_agm_rev1.gpickle")


# genero el espacio a explorar
# R0_g = np.linspace(0.1, 5, 5) # quiero explorar estos valores de R0
# T_r_g = np.linspace(10,30,5) # fijo el valor de tiempo de recuperación
# gamma_g = T_r_g ** (-1)
# beta_g = R0_g * gamma_g # las beta que me dan los R0 que quiero

R0_g = np.array([1.5, 2, 2.5, 3, 3.5, 4, 5, 7, 10])
#R0_g = np.array([2])
Tr_g = np.linspace(5,20, len(R0_g))
Tc_g = np.divide(Tr_g, R0_g)

omega_g = np.arange(0.01, 0.2, 0.05)

#countries_g = df['country_code'].sample(10).values # para simular solo un muestreo de los paises
countries_g = ['ESP','CHN','USA','ASM'] # para seleccionar paises especificos
#countries_g = ['ESP']
options = [countries_g, Tc_g, Tr_g, omega_g]

input_dataframe = []

full_options = []
for item in itertools.product(*options):
    full_options.append(item)
    
print("There are {} simulations".format(len(full_options)))

# este for me va generando todas las combinaciones posibles
for country, Tc, Tr, omega in tqdm(itertools.product(*options)):
        input_dataframe.append(sir_model(df, OD, Tc, Tr, omega, country, 1, 1000))
   

intial_country, idx_country, R0, Tc, Tr, omega, total_infected,\
            total_death, total_recovered   
   
columns = ["intial_country", "idx_country", "R0", "Tc", "Tr", "omega", "total_infected",\
            "total_death", "total_recovered"  ]
df_full = pd.DataFrame(input_dataframe,
                       columns = columns)

#df_full.to_pickle("/Users/agm/Documents/kschool/TFM/data_out/df_simulations.pkl")
df_full.to_pickle("/Users/agm/Documents/kschool/TFM/data_out/df_simulations_comp_rev5.pkl.zip")


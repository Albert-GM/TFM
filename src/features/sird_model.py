# adding directory to pythonpath for allow own functions
import sys
import os
import re
root_project = re.findall(r'(^\S*TFM)', os.getcwd())[0]
sys.path.append(root_project)

import pandas as pd
import numpy as np
from src.utils.sird_support import  infection_power, mortality_power,\
    top_k_countries, countries_reaction, closing_country, check_array_div



class SIRD_model:
    """
    Simulates the spread of a infectious disease around the world.
    Model based in the classic SIR model with deceased population. There is
    a SIRD model for each country and each one has its own population of susceptible,
    infected,recovered and deceased. The interactions are defined as movements
    of individuals from one country (origin) to another (destination). These
    individuals at the moment of moving can be susceptible, infected or recovered
    and they become part of the destination population. The model also enables
    countries to close the connection with other countries with a reaction time
    to do it.

    """

    df = pd.read_pickle(
        f"{root_project}/data/interim/country_info_final.pickle")
    OD = np.load(f"{root_project}/data/interim/od_matrix.npy")
    sim_time = 730
    initial_infected = 1
    # feat_time = 14 # number of days allowed to compute disease's features
    feat_time = 30
    def __init__(self, R0, Tr, omega, i_country, n_closed, react_time):

        self.R0 = R0 # reproduction number
        self.Tr = Tr # typical recovery time
        self.omega = omega # case fatility rate
        self.i_country = i_country
        self.n_closed = n_closed # countries in quarantine
        self.react_time = react_time # reaction time of countries to get into quarantine

    def simulate(self):
        """
        Begins the simulation with the parameters specified in the constructor

        Returns
        -------
        None.

        """

        self.idx_country = SIRD_model.df.loc[SIRD_model.df["country_code"]
                                             == self.i_country].index.item()
        self.N_i = SIRD_model.df['total_pop'].values  # Population of each country
        n = len(self.N_i)  # Total number of countries
        top_countries = top_k_countries(
            self.n_closed, SIRD_model.df, self.idx_country)
        # Start the SIRD matrix
        SIRD = np.zeros((n, 4))
        # Assign to the susceptible population all the population of the
        # country
        SIRD[:, 0] = self.N_i
        # Assign to the population of infected the initial number of infected and
        # subtracts it from the population of susceptible
        SIRD[self.idx_country, 1] += self.initial_infected
        SIRD[self.idx_country, 0] -= self.initial_infected
        # SIRD matrix normalized
        SIRD_p = SIRD / SIRD.sum(axis=1).reshape(-1, 1)
        
        N = self.N_i

        # Compute the epidemic's parameters of the SIRD model
        self.Tc = self.Tr / self.R0
        # Average number of contacts per person per time
        beta = self.Tc ** (-1)
        gamma = self.Tr ** (-1)  # Rate of recovered
        # R0 = beta / gamma
        # Make vectors from the parameters
        beta_v = np.full(n, beta)
        gamma_v = np.full(n, gamma)

        # Arrays to save the information from the simulation
        new_infected_t = np.zeros((n, SIRD_model.sim_time))
        new_recovered_t = np.zeros((n, SIRD_model.sim_time))
        new_deceased_t = np.zeros((n, SIRD_model.sim_time))
        SIRD_t = np.zeros((n, 4, SIRD_model.sim_time))

        self.OD_sim = SIRD_model.OD.copy()  # keep the original OD matrix

        flag_react = 1
        flag_deaths = 1

        # Start the simulation
        for t in range(SIRD_model.sim_time):
            # OD matrix of susceptible, infected, recovered and deceased
            S = np.array([SIRD_p[:, 0], ] * n).T
            I = np.array([SIRD_p[:, 1], ] * n).T
            R = np.array([SIRD_p[:, 2], ] * n).T
            D = np.array([SIRD_p[:, 3], ] * n).T
            # element-wise multiplication
            OD_S, OD_I, OD_R = np.floor(
                self.OD_sim * S), np.floor(self.OD_sim * I), np.floor(self.OD_sim * R)
            # People entering and leaving by group for each country
            out_S, in_S = OD_S.sum(axis=1), OD_S.sum(axis=0)
            out_I, in_I = OD_I.sum(axis=1), OD_I.sum(axis=0)
            out_R, in_R = OD_R.sum(axis=1), OD_R.sum(axis=0)
            # Update SIRD matrix according travels
            SIRD[:, 0] = SIRD[:, 0] - out_S + in_S
            SIRD[:, 1] = SIRD[:, 1] - out_I + in_I
            SIRD[:, 2] = SIRD[:, 2] - out_R + in_R
            # Cehck for negative values in SIRD
            SIRD = np.where(SIRD < 0, 0, SIRD)
            # Update population of each country
            N = SIRD.sum(axis=1)
            # Compute new infected at t.
            new_infected = check_array_div(beta_v * SIRD[:, 0] * SIRD[:, 1],
                                           N)
            # If the population N_i of a country is 0, new_infected is 0
            # New infected can't be higher than susceptible
            new_infected = np.where(new_infected > SIRD[:, 0], SIRD[:, 0],
                                    new_infected)
            # Compute recovered at t
            new_recovered = gamma_v * SIRD[:, 1]  # New recovered at t
            # Compute deceased at t
            new_deceased = self.omega * SIRD[:, 1]
            # Updating SIRD matrix according epidemic transitions
            SIRD[:, 0] = SIRD[:, 0] - new_infected # va sumando los decimales, pero abajo siempre hago floor..por eso dan valores diferentes
            SIRD[:, 1] = SIRD[:, 1] + new_infected - \
                new_recovered - new_deceased
            SIRD[:, 2] = SIRD[:, 2] + new_recovered
            SIRD[:, 3] = SIRD[:, 3] + new_deceased
            SIRD_p = check_array_div(SIRD, SIRD.sum(axis=1).reshape(-1, 1))
            # Saving information of the day t
            SIRD_t[:, :, t] = SIRD
            # new_infected_t[:, t] = new_infected
            # new_recovered_t[:, t] = new_recovered
            # new_deceased_t[:, t] = new_deceased
            
            # SIRD_t[:, :, t] = np.floor(SIRD)
            new_infected_t[:, t] = np.floor(new_infected)
            new_recovered_t[:, t] = np.floor(new_recovered)
            new_deceased_t[:, t] = np.floor(new_deceased)   
            
            deaths_t = np.sum(new_deceased_t, axis=0)
            deaths_total = np.sum(new_deceased_t)


            # Compute day first deceased and 2 weeks later
            if new_deceased_t.sum() > 1 and flag_deaths:
                if t <= SIRD_model.sim_time - (SIRD_model.feat_time + 1):
                    day_a, day_b = t, t + SIRD_model.feat_time
                # Assert that the interval is two weeks in extreme cases
                else:
                    day_a = SIRD_model.sim_time - (SIRD_model.feat_time + 1)
                    day_b = (SIRD_model.sim_time - 1)
                # if there is quarantine, close countries
                if self.n_closed > 0:
                    country_react, flag_react = countries_reaction(
                        day_a, self.react_time, top_countries)
                flag_deaths = 0 

            if not flag_react:
                self.OD_sim = closing_country(country_react, self.OD_sim, t)
        # If flag=1 there are not deceased
        if flag_deaths:
            day_a, day_b = 0, 0

        self.sim_results_ = {
            'new_infected_world_t': new_infected_t.sum(axis=0),
            'new_recovered_world_t': new_recovered_t.sum(axis=0),
            'new_deceased_world_t': new_deceased_t.sum(axis=0),
            'new_infected_t': new_infected_t,
            'new_recovered_t': new_recovered_t,
            'new_deceased_t': new_deceased_t,
            'SIRD_t': SIRD_t,
            'SIRD_p_t': check_array_div(SIRD_t,
                                        SIRD_t.sum(axis=1)[:, np.newaxis, :]),
            'SIRD_world_t': SIRD_t.sum(axis=0),
            'total_infected': new_infected_t.sum() + SIRD_model.initial_infected,
            'total_recovered': new_recovered_t.sum(),
            'total_deceased': new_deceased_t.sum(),
            'day_a': day_a,
            'day_b': day_b,
        }

        self.sim_results_['SIRD_world_p_t'] = self.sim_results_[
            'SIRD_world_t'] / self.sim_results_['SIRD_world_t'].sum(axis=0)

    def compute_disease_features(self):
        """
        Computes features of the epidemic from the data generated by the
        simulation.

        Returns
        -------
        None.

        """

        if not hasattr(self, 'sim_results_'):
            raise ValueError('The model needs to be simulated first.')

        if self.sim_results_['new_deceased_world_t'].sum() > 0:

            inf_pow_1, inf_pow_2, gradient_inf, \
                sum_gradient_inf, p_inf = infection_power(
                self.sim_results_['new_infected_world_t'],
                self.sim_results_['SIRD_world_t'],
                self.sim_results_['day_a'],
                self.sim_results_['day_b'],
                self.N_i[self.idx_country])

            mort_pow_1, mort_pow_2, mort_pow_3,\
                gradient_mort, sum_gradient_mort = mortality_power(
                    self.sim_results_['new_deceased_world_t'],
                    self.sim_results_['new_infected_world_t'],
                    self.sim_results_['SIRD_world_t'],
                    self.sim_results_['day_a'],
                    self.sim_results_['day_b'])
        else:
            inf_pow_1, inf_pow_2, gradient_inf,\
                sum_gradient_inf, p_inf = 0, 0, np.array([0]), 0, 0

            mort_pow_1, mort_pow_2, mort_pow_3, gradient_mort,\
                sum_gradient_mort = 0, 0, 0, np.array([0]), 0
                                                                                

        self.epidemic_features_ = {
            'inf_pow_1': inf_pow_1,
            'inf_pow_2': inf_pow_2,
            'gradient_inf': gradient_inf,
            'sum_gradient_inf': sum_gradient_inf,
            'p_inf': p_inf,
            'mort_pow_1': mort_pow_1,
            'mort_pow_2': mort_pow_2,
            'mort_pow_3': mort_pow_3,
            'gradient_mort': gradient_mort,
            'sum_gradient_mort': sum_gradient_mort
            
        }

    def get_simulation_data(self):
        """
        Returns simulation data and epidemic features. If output_mode='short'
        summarized data about the epidemic
        is provided, whereas output_mode='large', all the data is provided.

        Returns
        -------
        None.

        """

        if not hasattr(self, 'sim_results_'):
            raise ValueError('The model needs to be simulated first.')

        if not hasattr(self, 'epidemic_features_'):
            raise ValueError('The disease features need to be computed first.')

        self.sim_param = {
            'R0': self.R0,
            'Tr': self.Tr,
            'Tc': self.Tc,
            'omega': self.omega,
            'i_country': self.i_country,
            'idx_country': self.idx_country,
            'n_closed': self.n_closed,
            'react_time': self.react_time,
            'simulation_time': self.sim_time}

        return {
            **self.sim_param,
            **self.epidemic_features_,
            **self.sim_results_}

    @classmethod
    def update_params(cls, param_dict):
        """
        Updates de class attributtes.

        Parameters
        ----------

        param_dict : dict
            Dictionary containing the class attribute as key and the new attribute
            value as value. For example d = {'df': new_dataframe, 'OD': new_OD,
            'sim_time': new_sim_time, 'initial_infected': new_initial_infected}

        Returns
        -------
        None.

        """

        for item in param_dict.items():
            setattr(cls, item[0], item[1])


if __name__ == '__main__':
    # one simulation of the SIRD model with the spcified parameters
    R0 = 5
    Tr = 10
    omega = 0.3
    n_closed = 5
    react_time = 20
    i_country = 'ESP'
    sird_instance = SIRD_model(R0, Tr, omega, i_country,
                               n_closed, react_time)
    sird_instance.simulate()
    sird_instance.compute_disease_features()
    foo = sird_instance.get_simulation_data()

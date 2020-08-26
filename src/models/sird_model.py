# adding directory to pythonpath for allow own functions
import sys
import os
import re
root_project = re.findall(r'(^\S*TFM-master)', os.getcwd())[0]
sys.path.append(root_project)

from src.utils.sird_support import first_deceased, infection_power, mortality_power,\
    top_k_countries, countries_reaction, closing_country
import numpy as np
import pandas as pd
from scipy.stats import uniform, expon, randint
from sklearn.model_selection import ParameterSampler



class SIRD_model:
    """
    Simulates the spread of a contagious disease around the world.
    The main model is based on the SIR model, but adds other characteristics to
    make it more representative of how countries behave facing a global epidemic.
    The model simplifies the movement of people between countries to the movement
    between airports. The model incorporates the closure of airports with
    reaction time after a certain number of deceased from the epidemic is exceeded.

    """

    df = pd.read_pickle(
        f"{root_project}/data/interim/country_info_final.pickle")
    OD = np.load(f"{root_project}/data/interim/od_matrix.npy")
    sim_time = 730
    initial_infected = 1

    def __init__(self, R0, Tr, omega, i_country, limit_deceased, n_closed,
                 react_time):

        self.R0 = R0
        self.Tr = Tr
        self.omega = omega
        self.i_country = i_country
        self.limit_deceased = limit_deceased
        self.n_closed = n_closed
        self.react_time = react_time


    def simulate(self):
        """
        Begins the simulation with the parameters specified in the constructor

        Returns
        -------
        None.

        """
        
        self.idx_country = SIRD_model.df.loc[SIRD_model.df["country_code"]
                                            == self.i_country].index.item()
        N_i = SIRD_model.df['total_pop'].values  # Population of each country
        n = len(N_i)  # Total number of countries
        top_countries = top_k_countries(
            self.n_closed, SIRD_model.df, self.idx_country)
        # Start the SIRD matrix
        SIRD = np.zeros((n, 4))
        # Assign to the susceptible population all the population of the
        # country
        SIRD[:, 0] = N_i
        # Assign to the population of infected the initial number of infected and
        # subtracts it from the population of susceptible
        SIRD[self.idx_country, 1] += self.initial_infected
        SIRD[self.idx_country, 0] -= self.initial_infected
        SIRD_p = SIRD / SIRD.sum(axis=1).reshape(-1, 1)  # SIRD matrix normalized

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

        flag = 1  # Flag for countries_reaction function

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
            N_i = SIRD.sum(axis=1)
            # Compute new infected at t.
            new_infected = (beta_v * SIRD[:, 0] * SIRD[:, 1]) / N_i
            # If the population N_i of a country is 0, new_infected is 0
            new_infected = np.where(np.isnan(new_infected), 0, new_infected)
            # New infected can't be higher than susceptible
            new_infected = np.where(new_infected > SIRD[:, 0], SIRD[:, 0],
                                    new_infected)
            # Compute recovered at t
            new_recovered = gamma_v * SIRD[:, 1]  # New recovered at t
            # Compute deceased at t
            new_deceased = self.omega * SIRD[:, 1]
            # Updating SIRD matrix according epidemic transitions
            SIRD[:, 0] = SIRD[:, 0] - new_infected
            SIRD[:, 1] = SIRD[:, 1] + new_infected - new_recovered -new_deceased
            SIRD[:, 2] = SIRD[:, 2] + new_recovered
            SIRD[:, 3] = SIRD[:, 3] + new_deceased
            SIRD_p = SIRD / SIRD.sum(axis=1).reshape(-1, 1)
            # Checking for nan
            SIRD_p = np.where(np.isnan(SIRD_p), 0, SIRD_p)
            # Saving information of the day t
            SIRD_t[:, :, t] = np.floor(SIRD)
            new_infected_t[:, t] = np.floor(new_infected)
            new_recovered_t[:, t] = np.floor(new_recovered)
            new_deceased_t[:, t] = np.floor(new_deceased)

            if new_deceased.sum() > self.limit_deceased and flag:
                country_react, flag = countries_reaction(t, self.react_time,
                                                         top_countries)

            if not flag:
                self.OD_sim = closing_country(country_react, self.OD_sim, t)
                

        self.sim_results_ = {
            'new_infected_world_t': new_infected_t.sum(axis=0),
            'new_recovered_world_t': new_recovered_t.sum(axis=0),
            'new_deceased_world_t': new_deceased_t.sum(axis=0),
            'new_infected_t': new_infected_t,
            'new_recovered_t': new_recovered_t,
            'new_deceased_t': new_deceased_t,
            'SIRD_t': SIRD_t,
            'SIRD_p_t': SIRD_t / SIRD_t.sum(axis=1)[:, np.newaxis, :],
            'SIRD_world_t': SIRD_t.sum(axis=0),
            'total_infected': new_infected_t.sum() + SIRD_model.initial_infected,
            'total_recovered': new_recovered_t.sum(),
            'total_deceased': new_deceased_t.sum(),
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
            day_1, day_2 = first_deceased(self.sim_results_['new_deceased_world_t'])

            inf_pow_1, inf_pow_2, gradient_inf = infection_power(
                self.sim_results_['new_infected_world_t'],
                self.sim_results_['SIRD_world_t'],
                day_1,
                day_2)

            mort_pow_1, mort_pow_2, mort_pow_3,\
                gradient_mort = mortality_power(
                    self.sim_results_['new_deceased_world_t'],
                    self.sim_results_['new_infected_world_t'],
                    self.sim_results_['SIRD_world_t'],
                    day_1,
                    day_2)
        else:
            inf_pow_1, inf_pow_2,
            gradient_inf = 0, 0, np.array([0])

            mort_pow_1, mort_pow_2, mort_pow_3,
            gradient_mort = 0, 0, 0, np.array([0])

        
        self.epidemic_features_ = {
            
            'inf_pow_1': inf_pow_1,
            'inf_pow_2': inf_pow_2,
            'gradient_inf': gradient_inf,
            'mort_pow_1': mort_pow_1,
            'mort_pow_2': mort_pow_2,
            'mort_pow_3': mort_pow_3,
            'gradient_mort': gradient_mort,
            
            
            
            }


        # self.epidemic_features_ = {}
        # self.epidemic_features_['inf_pow_1'] = inf_pow_1
        # self.epidemic_features_['inf_pow_2'] = inf_pow_2
        # self.epidemic_features_['gradient_inf'] = gradient_inf
        # self.epidemic_features_['mort_pow_1'] = mort_pow_1
        # self.epidemic_features_['mort_pow_2'] = mort_pow_2
        # self.epidemic_features_['mort_pow_3'] = mort_pow_3
        # self.epidemic_features_['gradient_mort'] = gradient_mort

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
            'limit_deceased': self.limit_deceased,
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
            Dictionary containin the class attribute as key and the new attribute
            value as value. For example d = {'df': new_dataframe, 'OD': new_OD,
            'sim_time': new_sim_time, 'initial_infected': new_initial_infected}

        Returns
        -------
        None.

        """
        
        for item in param_dict.items():
            setattr(cls, item[0], item[1])
        
        
    
    @classmethod
    def update_countries(cls, df_countries):
        """
        Updates the class attribute with information about countries.

        Parameters
        ----------

        df_countries : pandas.DataFrame

        Returns
        -------
        None.

        """
        cls.df = df_countries
        
    @classmethod   
    def update_OD(cls, OD_matrix):
        """
        Updates the class attribute with information about origin-destination
        matrix.

        Parameters
        ----------

        OD_matrix : np.array
            If the number of countries in cls.df is N, this matris is NxN.

        Returns
        -------
        None.

        """
        
        cls.OD = OD_matrix


if __name__ == '__main__':

    sird_instance = SIRD_model(4, 15, 0.01, 'FRA', 100, 0, 5)
    sird_instance.simulate()
    sird_instance.compute_disease_features()
    foo = sird_instance.get_simulation_data()
    
    
    
    
    # def simulate(self):
    #     """
    #     Begins the simulation with the parameters specified in the constructor

    #     Returns
    #     -------
    #     None.

    #     """
        
    #     self.idx_country = SIR_model.df.loc[SIR_model.df["country_code"]
    #                                         == self.i_country].index.item()
    #     N_i = SIR_model.df['total_pop'].values  # Population of each country
    #     n = len(N_i)  # Total number of countries
    #     top_countries = top_k_countries(
    #         self.n_closed, SIR_model.df, self.idx_country)
    #     # Starting the SIR matrix
    #     SIR = np.zeros((n, 3))
    #     # Assign to the susceptible population all the population of the
    #     # country
    #     SIR[:, 0] = N_i
    #     # Assign to the population of infected the initial number of infected and
    #     # subtracts it from the population of susceptible
    #     SIR[self.idx_country, 1] += self.initial_infected
    #     SIR[self.idx_country, 0] -= self.initial_infected
    #     SIR_p = SIR / SIR.sum(axis=1).reshape(-1, 1)  # SIR matrix normalized

    #     # Compute the epidemic's parameters of the SIR model
    #     self.Tc = self.Tr / self.R0
    #     # Average number of contacts per person per time
    #     beta = self.Tc ** (-1)
    #     gamma = self.Tr ** (-1)  # Rate of removed
    #     # R0 = beta / gamma
    #     # Make vectors from the parameters
    #     beta_v = np.full(n, beta)
    #     gamma_v = np.full(n, gamma)

    #     # Arrays to save the information from the simulation
    #     new_infected_t = np.zeros((n, SIR_model.sim_time))
    #     new_removed_t = np.zeros((n, SIR_model.sim_time))
    #     deaths_t = np.zeros((n, SIR_model.sim_time))
    #     SIR_t = np.zeros((n, 3, SIR_model.sim_time))

    #     self.OD_sim = SIR_model.OD.copy()  # keep the original OD matrix

    #     flag = 1  # Flag for countries_reaction function

    #     # Starting the simulation
    #     for t in range(SIR_model.sim_time):
    #         # OD matrix of susceptible, infected and removed
    #         S = np.array([SIR_p[:, 0], ] * n).T
    #         I = np.array([SIR_p[:, 1], ] * n).T
    #         R = np.array([SIR_p[:, 2], ] * n).T
    #         # element-wise multiplication
    #         OD_S, OD_I, OD_R = np.floor(
    #             self.OD_sim * S), np.floor(self.OD_sim * I), np.floor(self.OD_sim * R)
    #         # People entering and leaving by group for each country
    #         out_S, in_S = OD_S.sum(axis=1), OD_S.sum(axis=0)
    #         out_I, in_I = OD_I.sum(axis=1), OD_I.sum(axis=0)
    #         out_R, in_R = OD_R.sum(axis=1), OD_R.sum(axis=0)
    #         # Updating SIR matrix according travels
    #         SIR[:, 0] = SIR[:, 0] - out_S + in_S
    #         SIR[:, 1] = SIR[:, 1] - out_I + in_I
    #         SIR[:, 2] = SIR[:, 2] - out_R + in_R
    #         # Checking for negative values in SIR
    #         SIR = np.where(SIR < 0, 0, SIR)
    #         # Updating population of each country
    #         N_i = SIR.sum(axis=1)
    #         # Computing new infected people at t.
    #         new_infected = (beta_v * SIR[:, 0] * SIR[:, 1]) / N_i
    #         # If the population N_i of a country is 0, new_infected is 0
    #         new_infected = np.where(np.isnan(new_infected), 0, new_infected)
    #         # New infected can't be higher than susceptible
    #         new_infected = np.where(new_infected > SIR[:, 0], SIR[:, 0],
    #                                 new_infected)
    #         new_removed = gamma_v * SIR[:, 1]  # New removed at t
    #         # deaths computed from removed people at period t
    #         deaths = new_removed * self.omega
    #         # Updating SIR matrix according epidemic transitions
    #         SIR[:, 0] = SIR[:, 0] - new_infected
    #         SIR[:, 1] = SIR[:, 1] + new_infected - new_removed
    #         SIR[:, 2] = SIR[:, 2] + new_removed
    #         SIR_p = SIR / SIR.sum(axis=1).reshape(-1, 1)
    #         # Checking for nan
    #         SIR_p = np.where(np.isnan(SIR_p), 0, SIR_p)
    #         # Saving information of the day t
    #         SIR_t[:, :, t] = np.floor(SIR)
    #         new_infected_t[:, t] = np.floor(new_infected)
    #         new_removed_t[:, t] = np.floor(new_removed)
    #         deaths_t[:, t] = np.floor(deaths)

    #         if deaths_t.sum() > self.limit_deaths and flag:
    #             country_react, flag = countries_reaction(t, self.react_time,
    #                                                      top_countries)

    #         if not flag:
    #             self.OD_sim = closing_country(country_react, self.OD_sim, t)
                

    #     self.sim_results_ = {
    #         'new_infected_world_t': new_infected_t.sum(axis=0),
    #         'new_removed_world_t': new_removed_t.sum(axis=0),
    #         'deaths_world_t': deaths_t.sum(axis=0),
    #         'SIR_world_t': SIR_t.sum(axis=0),
    #         'SIR_p_t': SIR_t / SIR_t.sum(axis=1)[:, np.newaxis, :],
    #         'total_infected': new_infected_t.sum() + SIR_model.initial_infected,
    #         'total_removed': new_removed_t.sum(),
    #         'total_death': deaths_t.sum(),
    #         'SIR_t': SIR_t,
    #         'new_infected_t': new_infected_t,
    #         'new_removed_t': new_removed_t,
    #         'deaths_t': deaths_t,
    #         }
        
    #     self.sim_results_['SIR_world_p_t'] = self.sim_results_[
    #         'SIR_world_t'] / self.sim_results_['SIR_world_t'].sum(axis=0)
    
    
    
    
    
    
        # self.sim_results_ = {}
        # self.sim_results_['new_infected_world_t'] = new_infected_t.sum(axis=0)
        # self.sim_results_['new_removed_world_t'] = new_removed_t.sum(axis=0)
        # self.sim_results_['deaths_world_t'] = deaths_t.sum(axis=0)
        # self.sim_results_['SIR_world_t'] = SIR_t.sum(axis=0)
        # self.sim_results_['SIR_world_p_t'] = self.sim_results_[
        #     'SIR_world_t'] / self.sim_results_['SIR_world_t'].sum(axis=0)
        # self.sim_results_['SIR_p_t'] = SIR_t / \
        #     SIR_t.sum(axis=1)[:, np.newaxis, :]
        # self.sim_results_['total_infected'] = new_infected_t.sum(
        # ) + SIR_model.initial_infected
        # self.sim_results_['total_removed'] = new_removed_t.sum()
        # self.sim_results_['total_death'] = deaths_t.sum()
        # self.sim_results_['SIR_t'] = SIR_t
        # self.sim_results_['new_infected_t'] = new_infected_t
        # self.sim_results_['new_removed_t'] = new_removed_t
        # self.sim_results_['deaths_t'] = deaths_t
        
        # self.new_infected_world_t_ = self.new_infected_t_.sum(axis=0)
        # self.new_removed_world_t_ = self.new_removed_t_.sum(axis=0)
        # self.deaths_world_t_ = self.deaths_t_.sum(axis=0)
        # self.SIR_world_t_ = self.SIR_t_.sum(axis=0)
        # self.SIR_world_p_t_ = self.SIR_world_t_ / self.SIR_world_t_.sum(axis=0)
        # self.SIR_p_t_ = self.SIR_t_ / self.SIR_t_.sum(axis=1)[:, np.newaxis, :]
        # self.total_infected_ = self.new_infected_t_.sum() + SIR_model.initial_infected
        # self.total_removed_ = self.new_removed_t_.sum()
        # self.total_death_ = self.deaths_t_.sum()
    
    
    # def get_simulation_data(self, output_mode='short'):
    #     """
    #     Returns simulation data and epidemic features. If output_mode='short'
    #     summarized data about the epidemic
    #     is provided, whereas output_mode='large', all the data is provided.

    #     Parameters
    #     ----------
    #     output_mode : string, optional
    #         'short' or 'large'. The default is 'short'.

    #     Raises
    #     ------
    #     ValueError
    #         Raise an error if the output_mode value is not correct.

    #     Returns
    #     -------
    #     tuple

    #     """

    #     output_mode_options = ['short', 'large']

    #     if output_mode not in output_mode_options:
    #         raise ValueError(
    #             f"Invalid argument. Expected one of {output_mode_options}")

    #     elif output_mode == 'short':
    #         return self.i_country, self.idx_country, self.R0, self.Tc, self.Tr,\
    #             self.omega, self.inf_pow_1, self.inf_pow_2,\
    #             self.mort_pow_1, self.mort_pow_2, self.mort_pow_3,\
    #             self.limit_deaths, self.n_closed,\
    #             self.react_time, self.total_infected_, self.total_death_,\
    #             self.total_death_, self.total_removed_

    #     elif output_mode == 'large':
    #         return self.i_country, self.idx_country, self.R0, self.Tc, self.Tr,\
    #             self.omega, self.inf_pow_1, self.inf_pow_2, self.gradient_inf,\
    #             self.mort_pow_1, self.mort_pow_2, self.mort_pow_3,\
    #             self.gradient_mort, self.limit_deaths, self.n_closed,\
    #             self.react_time, self.total_infected_, self.total_death_,\
    #             self.total_death_, self.total_removed_, self.new_infected_t_,\
    #             self.new_infected_world_t_, self.deaths_t_, self.deaths_world_t_,\
    #             self.new_removed_t_, self.new_infected_world_t_, self.SIR_t_,\
    #             self.SIR_world_t_, self.SIR_p_t_, self.SIR_world_p_t_

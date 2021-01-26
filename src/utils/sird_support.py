# =============================================================================
# Functions to use in some computations of the SIRD model 
# =============================================================================

import numpy as np
from scipy.stats import truncexpon


def top_k_countries(n_closed, df, idx_initial):
    """
    Computes the top k countries with more traffic of tourism (arrivals+departures)
    where k=n_closed, which indicates how many countries to close.

    Parameters
    ----------
    n_closed : int
        Countries to close.
    df : pandas.DataFrame
    idx_initial : int
        Index of the country when the epidemic started.

    Returns
    -------
    idx_countries : list
        List with the index of the countries to close.

    """

    traffic = df['arrivals'] + df['departures']
    idx_countries = list(traffic.nlargest(n=n_closed).index)
    # if idx_initial not in idx_countries and n_closed != 0:
    #     idx_countries.append(idx_initial)

    return idx_countries

def countries_reaction(t, react_time, top_countries):
    """
    Computes how long a country takes to react once the deceased limit is exceeded. 

    Parameters
    ----------
    t : int
        Simulation instant.
    react_time : int
        Parameter of the exponential distribution.
    top_countries : list
        List with the index of the countries to close.

    Returns
    -------
    country_react : dictionary
        Reaction time of each country in top_countries.
    flag : int

    """

    country_react = {}
    # print(t)
    for country in top_countries:
        delay = truncexpon.rvs(loc=react_time, b=10, size=1).astype('int')
        # print(f"{country} reaction {delay}")
        country_react[country] = t + delay
        # country_react[country] = t + truncexpon.rvs(
        #     loc=react_time, b=react_time + 30, size=1).astype('int')
        # country_react[country] = t + react_time
    flag = 0

    return country_react, flag

def closing_country(country_react, OD, t):
    """
    Closes the country according to its reaction time.

    Parameters
    ----------
    country_react : dict
        Reaction time of each country in top_countries.
    OD : np.array
        Data origin-destination between countries.
    t : int
        Simulation instant.

    Returns
    -------
    OD : np.array
        Data origin-destination between countries modificated.

    """

    for country_idx, time in country_react.items():
        if time == t:
            OD[country_idx, :] = 0
            OD[:, country_idx] = 0
            # print(f"closing {country_idx} at {t}")

    return OD

def first_deceased(new_deceased_world_t, period=14):
    """
    Computes the first deceased that occurs in the pandemic and returns that day
    and the day for which period days have passed

    Parameters
    ----------
    new_deceased_world_t : np.array
    period : int, optional

    Returns
    -------
    day_1 : float
        Day of first deceased.
    day_2 : TYPE
        Day of first deceased plus period.

    """

    day_1 = np.argwhere(new_deceased_world_t > 0)[0, 0]
    day_2 = day_1 + period
    if day_2 > new_deceased_world_t.shape[0] - 1:
        day_2 = new_deceased_world_t.shape[0] - 1 

    return day_1, day_2

def check_division(n, d):
    """
    Returns 0 if denominator is 0
    """

    return n / d if d else 0

def check_array_div(n, d):
    """
    Returns 0 if denominator is 0, array mode.
    """
    
    return np.divide(n, d, out=np.zeros_like(n), where=d!=0)

def infection_power(new_infected_world_t, SIRD_world_t, day_1, day_2, N):
    """
    Computes some parameters related with the infecting power of the disease.

    Parameters
    ----------
    new_infected_world_t : np.array
    SIRD_world_t : np.array
    day_1 : float
    day_2 : float

    """
    
    if day_2 == 0:
        ind_inf_1, ind_inf_2, ind_inf_grad, ind_inf_sumgrad, ind_inf_p = 0, 0, 0, 0, 0 # there have been no deceased
    else:
        ind_inf_1 = new_infected_world_t[day_1:day_2].sum() / (day_2 - day_1)
        
        ind_inf_2 = check_division((SIRD_world_t[1, day_2] -
                   SIRD_world_t[1, day_1]),  SIRD_world_t[1, day_1])
        #slope_2 could be negative is number of infected decrease
        ind_inf_grad = np.gradient(SIRD_world_t[1, day_1:day_2])
        ind_inf_sumgrad = ind_inf_grad.sum()
        
        ind_inf_p = new_infected_world_t[day_1:day_2].sum() / N
        
        
    return ind_inf_1, ind_inf_2, ind_inf_grad, ind_inf_sumgrad, ind_inf_p

def mortality_power(new_deceased_world_t, new_infected_world_t, SIRD_world_t,
                    day_1, day_2):
    """
    Computes some parameters related with the mortality power of the disease.

    Parameters
    ----------
    new_deceased_world_t : np.array
    new_infected_world_t : np.array
    SIRD_world_t : np.array
    day_1 : float
    day_2 : float

    """
    if day_2 == 0:
        ind_mort_1, ind_mort_2, ind_mort_3, ind_mort_grad, ind_mort_sumgrad = 0, 0, 0, 0, 0
    else:
        ind_mort_1 = check_division(new_deceased_world_t[:day_2].sum(),
                                new_infected_world_t[day_1:day_2].sum())
        
        ind_mort_2 = check_division(new_deceased_world_t[day_1:day_2].sum(),
                                 SIRD_world_t[1, day_2] - SIRD_world_t[1, day_1])
        
        ind_mort_3 = check_division(new_deceased_world_t[day_1:day_2].sum(),
                                 SIRD_world_t[2, day_2] - SIRD_world_t[2, day_1])

    
        ind_mort_grad = np.gradient(new_deceased_world_t[day_1:day_2])
        ind_mort_sumgrad = ind_mort_grad.sum()

    return ind_mort_1, ind_mort_2, ind_mort_3, ind_mort_grad, ind_mort_sumgrad

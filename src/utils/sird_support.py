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
    if idx_initial not in idx_countries and n_closed != 0:
        idx_countries.append(idx_initial)

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
    for country in top_countries:
        # country_react[country] = np.random.exponential(scale=2,
        #     size=1).astype('int') + react_time + t
        country_react[country] = t + truncexpon.rvs(
            loc=react_time, b=react_time + 30, size=1).astype('int')
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

    Returns
    -------
    slope_1 : float
    slope_2 : float
    gradient : np.array

    """
    
    if day_2 == 0:
        ratio_1, ratio_2, gradient, sum_gradient, p_inf = 0, 0, 0, 0, 0 # there have been no deceased
    else:
        ratio_1 = new_infected_world_t[day_1:day_2].sum() / (day_2 - day_1)
        
        ratio_2 = check_division((SIRD_world_t[1, day_2] -
                   SIRD_world_t[1, day_1]),  SIRD_world_t[1, day_1])
        #slope_2 could be negative is number of infected decrease
        gradient = np.gradient(SIRD_world_t[1, day_1:day_2])
        sum_gradient = gradient.sum()
        
        p_inf = new_infected_world_t[day_1:day_2].sum() / N
        
        
    return ratio_1, ratio_2, gradient, sum_gradient, p_inf

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

    Returns
    -------
    ratio_1 : float
    ratio_2 : float
    ratio_3 : float
    gradient : np.array

    """
    if day_2 == 0:
        ratio_1, ratio_2, ratio_3, gradient, sum_gradient = 0, 0, 0, 0, 0
    else:
        ratio_1 = check_division(new_deceased_world_t[:day_2].sum(),
                                new_infected_world_t[day_1:day_2].sum())
        
        ratio_2 = check_division(new_deceased_world_t[day_1:day_2].sum(),
                                 SIRD_world_t[1, day_2] - SIRD_world_t[1, day_1])
        
        ratio_3 = check_division(new_deceased_world_t[day_1:day_2].sum(),
                                 SIRD_world_t[2, day_2] - SIRD_world_t[2, day_1])

    
        gradient = np.gradient(new_deceased_world_t[day_1:day_2])
        sum_gradient = gradient.sum()

    return ratio_1, ratio_2, ratio_3, gradient, sum_gradient

# =============================================================================
# Model that simulates the spread of a contagious disease around the world.
# The main model is based on the SIR model, but adds other characteristics to
# make it more representative of how countries behave facing a global epidemic.
# The model simplifies the movement of people between countries to the movement
# between airports. 
# The model incorporates the closure of airports with reaction time after a
# certain number of deaths from the epidemic is exceeded.
# =============================================================================


import numpy as np


def first_death(deaths_global_t, period=14):

    day_1 = np.argwhere(deaths_global_t > 0)[0, 0]
    day_2 = day_1 + period

    return day_1, day_2


def infection_power(new_infected_global_t, SIR_global_t, day_1, day_2):

    slope_1 = new_infected_global_t[day_1:day_2].sum() / (day_2 - day_1)
    
    slope_2 = (SIR_global_t[1, day_2] -
               SIR_global_t[1, day_1]) / SIR_global_t[1, day_1]
    
    gradient = np.gradient(SIR_global_t[1, day_1:day_2])

    return slope_1, slope_2, gradient


def mortality_power(deaths_global_t, new_infected_global_t, SIR_global_t,
                    day_1, day_2):

    ratio_1 = deaths_global_t[:day_2].sum(
    ) / (new_infected_global_t[day_1:day_2].sum())
    
    ratio_2 = (deaths_global_t[day_1:day_2].sum() /
               (SIR_global_t[1, day_2] - SIR_global_t[1, day_1]))
    
    ratio_3 = (deaths_global_t[day_1:day_2].sum() /
               (SIR_global_t[2, day_2] - SIR_global_t[2, day_1]))
    
    gradient =  np.gradient(deaths_global_t[day_1:day_2])
    
    return ratio_1, ratio_2, ratio_3, gradient


def top_k_countries(n_closed, df, idx_initial):

    traffic = df['arrivals'] + df['departures']
    idx_countries = list(traffic.nlargest(n=n_closed).index)
    if idx_initial not in idx_countries and n_closed != 0:
        idx_countries.append(idx_initial)

    return idx_countries


def countries_reaction(t, react_time, top_countries):

    country_react = {}
    for country in top_countries:
        country_react[country] = np.random.exponential(size=1).astype('int')\
            + react_time + t

    flag = 0

    return country_react, flag


def closing_country(country_react, OD_sim, t):

    for country_idx, time in country_react.items():
        if time == t:
            OD_sim[country_idx, :] = 0
            OD_sim[:, country_idx] = 0
            # print(f"closing {country_idx} at {t}")

    return OD_sim


def sir_model(
        df,
        OD,
        R0,
        Tr,
        omega,
        initial_country,
        initial_infected=1,
        limit_deaths=100,
        n_closed=5,
        react_time=5,
        T=730,
        output_mode=0):
    """
    Simulation of a SIR model for each country with interactions with other
    countries according to the origin-destination matrix provided that
    represents the movement of people between countries. Initally all
    the population is susceptible and the disease begins in one country.

    Params
    -----------
    df: Dataframe
        Dataframe with information about countries
    OD: numpy.array (n,n)
        Matrix of origin-destination, where i is the country of origin and j is
        the country of destination
    R0: float
        Basic reproduction number, expected number of cases directly generated
        by one case
    Tr: float
        Recovery time
    omega: float
        Disease's death rate
    initial_country: string
        Country where disease begins
    initial_infected: int
        Number of initial infected
    T: int
        Number of days to simulate
    output_mode: int
        0 by default (brief output), 1 if complete output is desired

    Return
    -------------
    steady: bool
        True if the epidemic has reached an equilibrium state, False otherwise
    index_country: int
        Index of the country where the epidemic began
    R0: float
        Epidemic's reproduction number
    total_infected: int
        Total number of infected at the end of the epidemic
    total_death: int
        Total number of deaths at the end of the epidemic
    new_infected_global_t: np.array (T)
        Global new infections at each instant t
    new_infected_t: np.array (n,T)
        New infections by country at each instant t
    deaths_global_t: np.array (T)
        Global deaths at each instant t
    deaths_t: np.array (n,T)
        Deaths by country at each instant t
    SIR_global_t: np.array (3,T)
        Global susceptible, infected, removed at each instant t
    SIR_global_p_t : np.array (3,T)
        Global proportion of Susceptible, infected, removed at each instant t
    SIR_t: np.array (3,n,T)
        Susceptible, infected, removed by country at each instant t
    SIR_p_t: np.array (3,n,T)
        Proportion of susceptible, infected, removed by country at each instant t
    days_1: int
        Days between the first death of the epidemic and the second one
    deaths_info: np.array
        Important information about deaths
    """
    print(OD.sum(), "principio funcion\n=======")
    idx_country = df.loc[df["country_code"] == initial_country].index.item()
    N_i = df['total_pop'].values  # Population of each country
    n = len(N_i)  # Total number of countries
    top_countries = top_k_countries(n_closed, df, idx_country)
    # Starting the SIR matrix
    SIR = np.zeros((n, 3))
    # Assign to the susceptible population all the population of the country
    SIR[:, 0] = N_i
    # Assign to the population of infected the initial number of infected and
    # subtracts it from the population of susceptible
    SIR[idx_country, 1] += initial_infected
    SIR[idx_country, 0] -= initial_infected
    SIR_p = SIR / SIR.sum(axis=1).reshape(-1, 1)  # SIR matrix normalized

    # Compute the epidemic's parameters of the SIR model
    Tc = Tr / R0
    beta = Tc ** (-1)  # Average number of contacts per person per time
    gamma = Tr ** (-1)  # Rate of removed
    # R0 = beta / gamma
    # Make vectors from the parameters
    beta_v = np.full(n, beta)
    gamma_v = np.full(n, gamma)

    # Arrays to save the information from the simulation
    new_infected_t = np.zeros((n, T))
    new_removed_t = np.zeros((n, T))
    deaths_t = np.zeros((n, T))
    SIR_t = np.zeros((n, 3, T))

    flag = 1  # Flag for countries_reaction function
    
    OD_sim = OD.copy()

    # Starting the simulation
    for t in range(T):
        # OD matrix of susceptible, infected and removed
        S = np.array([SIR_p[:, 0], ] * n).T
        I = np.array([SIR_p[:, 1], ] * n).T
        R = np.array([SIR_p[:, 2], ] * n).T
        # element-wise multiplication
        OD_S, OD_I, OD_R = np.floor(OD_sim * S), np.floor(OD_sim * I), np.floor(OD_sim * R)
        # People entering and leaving by group for each country
        out_S, in_S = OD_S.sum(axis=1), OD_S.sum(axis=0)
        out_I, in_I = OD_I.sum(axis=1), OD_I.sum(axis=0)
        out_R, in_R = OD_R.sum(axis=1), OD_R.sum(axis=0)
        # Updating SIR matrix according travels
        SIR[:, 0] = SIR[:, 0] - out_S + in_S
        SIR[:, 1] = SIR[:, 1] - out_I + in_I
        SIR[:, 2] = SIR[:, 2] - out_R + in_R
        # Checking for negative values in SIR
        SIR = np.where(SIR < 0, 0, SIR)
        # Updating population of each country
        N_i = SIR.sum(axis=1)
        # Computing new infected people at t.
        new_infected = (beta_v * SIR[:, 0] * SIR[:, 1]) / N_i
        # If the population N_i of a country is 0, new_infected is 0
        new_infected = np.where(np.isnan(new_infected), 0, new_infected)
        # New infected can't be higher than susceptible
        new_infected = np.where(new_infected > SIR[:, 0], SIR[:, 0],
                                new_infected)
        new_removed = gamma_v * SIR[:, 1]  # New removed at t
        # deaths computed from removed people at period t
        deaths = new_removed * omega
        # Updating SIR matrix according epidemic transitions
        SIR[:, 0] = SIR[:, 0] - new_infected
        SIR[:, 1] = SIR[:, 1] + new_infected - new_removed
        SIR[:, 2] = SIR[:, 2] + new_removed
        SIR_p = SIR / SIR.sum(axis=1).reshape(-1, 1)
        # Checking for nan
        SIR_p = np.where(np.isnan(SIR_p), 0, SIR_p)
        # Saving information of the day t
        SIR_t[:, :, t] = np.floor(SIR)
        new_infected_t[:, t] = np.floor(new_infected)
        new_removed_t[:, t] = np.floor(new_removed)
        deaths_t[:, t] = np.floor(deaths)

        if deaths_t.sum() > limit_deaths and flag:
            country_react, flag = countries_reaction(t, react_time,
                                                     top_countries)

        if not flag:
            OD_sim = closing_country(country_react, OD_sim, t)

    new_infected_global_t = new_infected_t.sum(axis=0)
    new_removed_global_t = new_removed_t.sum(axis=0)
    deaths_global_t = deaths_t.sum(axis=0)
    SIR_global_t = SIR_t.sum(axis=0)
    SIR_global_p_t = SIR_global_t / SIR_global_t.sum(axis=0)
    SIR_p_t = SIR_t / SIR_t.sum(axis=1)[:, np.newaxis, :]
    total_infected = new_infected_t.sum() + initial_infected
    total_removed = new_removed_t.sum()
    total_death = deaths_t.sum()

    if deaths_global_t.sum() > 0:
        day_1, day_2 = first_death(deaths_global_t)
        inf_pow_1, inf_pow_2, gradient_inf = infection_power(
            new_infected_global_t, SIR_global_t, day_1, day_2)
        mort_pow_1, mort_pow_2, mort_pow_3, gradient_mort = mortality_power(
            deaths_global_t, new_infected_global_t, SIR_global_t, day_1, day_2)
    else:
        inf_pow_1, inf_pow_2, gradient_inf = 0, 0, np.array([0])
        mort_pow_1, mort_pow_2, mort_pow_3, gradient_mort = 0, 0, 0, np.array([0])

    if output_mode == 1:
        output = initial_country, idx_country, R0, Tc, Tr, omega, inf_pow_1, inf_pow_2, gradient_inf,\
            mort_pow_1, mort_pow_2, mort_pow_3, gradient_mort, limit_deaths, n_closed, react_time, total_infected,\
            total_death, total_removed, new_infected_t,\
            new_infected_global_t, deaths_t, deaths_global_t,\
            new_removed_t, new_removed_global_t, SIR_t, SIR_global_t,\
            SIR_p_t, SIR_global_p_t
    else:
        output = initial_country, idx_country, R0, Tc, Tr, omega, inf_pow_1, inf_pow_2, gradient_inf,\
            mort_pow_1, mort_pow_2, mort_pow_3, gradient_mort, limit_deaths, n_closed, react_time, total_infected,\
            total_death, total_removed
            
    print(OD.sum(), "final de la funcion\n=======")
    return output




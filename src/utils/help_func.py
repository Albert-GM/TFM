import pandas as pd
import numpy as np
import networkx as nx
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def extract_indicator(df, indicator, initial_year=None, final_year=None):
    """
    Function that allows to select the indicator and the years or year we are
    interested about. If intial year and final year is passed, the function
    returns the information from intial year to final year.If not initial year
    and final year is passed, the function returns the information of all the
    years. If final year is not passed, the function return the information
    only of one year (initial year).

    Parameters
    ---------
    df : dataframe
        Dataframe containing the world indicators by country and year
    indicator : string
        Name of the indicador we want to exctract the information
    initial_year : int
    final_year : in

    Returns
    --------
    Dataframe with the information about the indicador with the country name as row index and years as columns

    """

    if initial_year is None and final_year is None:
        new_df = df.loc[df['Indicator Name'] == indicator].drop(
            ['Indicator Name', 'Indicator Code'], axis=1)
        return new_df.reset_index(drop=True)
    elif final_year is None:
        keep_columns = ['Country Name', 'Country Code', str(initial_year)]
        new_df = df.loc[df['Indicator Name'] == indicator, keep_columns]
        new_df.rename(columns={str(initial_year): indicator}, inplace=True)
        return new_df.reset_index(drop=True)
    else:
        keep_columns = ['Country Name', 'Country Code'] + \
            list(range(initial_year, final_year + 1))
        keep_columns = [str(x) for x in keep_columns]
        new_df = df.loc[df['Indicator Name'] == indicator, keep_columns]
        return new_df.reset_index(drop=True)


def add_countryandloc(df, df_continents, df_coord):
    """
    Add to a dataframe columns with the Continent Name, Continent Code;
    Longitude and Latitude of the country

    """
    df_continents.rename(
        columns={
            'Three_Letter_Country_Code': 'Country Code'},
        inplace=True)
    df_continents = df_continents.loc[:, [
        'Continent_Name', 'Continent_Code', 'Country Code']]
    df = pd.merge(df, df_continents, on='Country Code', how='inner')
    df = pd.merge(df, df_coord, on='Country Code', how='inner')
    return df


def last_values(df):
    """
    It returns a dataframe with the last values available by country. The
    dataframe has country as index and years as columns. First is necessary to
    apply extract_indicator to the original dataframe from world_indicators.

    """
    years = list(range(1960, 2020))
    years = [str(x) for x in years]

    last_values = []
    # flag = False
    # mejorar este codigo, puedo añadir una flag, para decirme si todos han
    # sido nan
    for country in df.iterrows():
        s = country[1]
        for year in reversed(years):
            # print(s[year])
            if not pd.isnull(s[year]):
                last_values.append(s[year])
                flag = True
                break
            elif year == '1960':
                last_values.append(np.nan)
            else:
                pass
    df2 = df.copy()  # para no cambiar el df del input
    df2['last_value'] = last_values
    return df2[['Country Name', 'Country Code', 'last_value']]


def results_searchcv(estimator, X_test=None, y_test=None):
    """
    Prints out useful information about a trained GridsearchCV object or a
    RandomizedSearchCV object from the sklearn library. Given a pair X_test,
    y_test prints out the score and the mean absolute error from the estimator.

    Parameters
    ----------
    estimator : sklearn.model_selection._search.GridSearchCV
        A trained estimator.
    X_test : dataframe or array
    y_test : dataframe series or array

    Returns
    -------
    None.

    """

    print(f"The best score is:\n{estimator.best_score_}")
    print(f"The best parameters found are:\n{estimator.best_params_}")
    if X_test is not None and y_test is not None:
        print(f"The score in test is:\n{estimator.score(X_test, y_test)}")
        y_predicted = estimator.predict(X_test)
        print(f"The r2-square is\n{r2_score(y_test, y_predicted)}")
        print(f"The MAE is:\n{mean_absolute_error(y_test, y_predicted)}")
    print("==============")


def construct_dataframe(l, output_mode=0):
    """
    Makes a dataframe with the output of the sir_model function dependeding on
    the output_mode variable

    Parameters
    ----------
    l : list
        List of tuples, output of the function sir_model.
    output_mode : int
        0 by default (brief output), 1 if complete output is desired

    Returns
    -------
    Dataframe

    """
    # if output_mode == 0:
    #     columns = [
    #         'initial_country',
    #         'idx_country',
    #         'R0',
    #         'Tc',
    #         'Tr',
    #         'omega',
    #         'total_infected',
    #         'total_death',
    #         'total_recovered']
        
    if output_mode == 0:
        columns = [
            'initial_country',
            'idx_country',
            'R0',
            'Tc',
            'Tr',
            'omega',
            'inf_power_1',
            'inf_power_2',
            'gradient_inf',
            'mort_power_1',
            'mort_power_2',
            'mort_power_3',
            'gradient_mort',
            'limit_deaths',
            'n_closed',
            'react_time',
            'total_infected',
            'total_death',
            'total_recovered']
    else:
        columns = [
            'initial_country',
            'idx_country',
            'R0',
            'Tc',
            'Tr',
            'omega',
            'inf_power_1',
            'inf_power_2',
            'gradient_inf',
            'mort_power_1',
            'mort_power_2',
            'mort_power_3',
            'gradient_mort',
            'limit_deaths',
            'n_closed',
            'react_time',
            'total_infected',
            'total_death',
            'total_recovered',
            'new_infected_t',
            'new_infected_global_t',
            'deaths_t',
            'deaths_global_t',
            'new_recovered_t',
            'new_recovered_global_t',
            'SIR_t',
            'SIR_global_t',
            'SIR_p_t',
            'SIR_global_p_t']

    return pd.DataFrame(l, columns=columns)

def top_k_connected(df, k):
    
    graph = nx.read_gpickle('../../data/interim/routes_countries.gpickle')

    with open('../../data/interim/alpha3_to_alpha2.txt', 'r') as file:
        alpha3_to_alpha2 = json.load(file)

    degree = dict(nx.degree_centrality(graph))
    df['iso2'] = df['country_code'].map(alpha3_to_alpha2)
    df['degree'] = df['iso2'].map(degree)
    
    return df.sort_values(by='degree', ascending=False)['country_code'].iloc[:k].tolist()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

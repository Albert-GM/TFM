# =============================================================================
# Own functions that facilitate repetitive actions
# =============================================================================


import pandas as pd
import numpy as np
import networkx as nx
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import os
import re
root_project = re.findall(r'(^\S*TFM-master)', os.getcwd())[0]


def extract_indicator(df, indicator, initial_year=None, final_year=None):
    """
    Selects the indicator and the years or year we are
    interested about. If intial year and final year is passed, the function
    returns the information from intial year to final year.If not initial year
    and final year is passed, the function returns the information of all the
    years. If final year is not passed, the function return the information
    only of one year (initial year).

    Parameters
    ---------
    df : pandas.DataFrame
        Dataframe containing the world indicators by country and year
    indicator : string
        Name of the indicador we want to exctract the information
    initial_year : int
    final_year : in

    Returns
    --------
    pandas.DataFrame
        Dataframe with the information about the indicador with the country
        name as row index and years as columns

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
    Adds to a dataframe columns with the Continent Name, Continent Code;
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
    Provides a dataframe with the last values available by country. The
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
    X_test : pandas.DataFrame or array
    y_test : pandas.DataFrame, pandas.Series or array

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


def construct_dataframe(l):
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
    pandas.DataFrame

    """

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
    """
    Provides the k countries with higher degree

    Parameters
    ----------
    df : pandas.DataFrame
    k : int
        Number of countries to return

    Returns
    -------
    pandas.DataFrame
        k countries with higher degree.

    """

    graph = nx.read_gpickle(
        f"{root_project}/data/interim/routes_countries.gpickle")

    with open(f"{root_project}/data/interim/alpha3_to_alpha2.txt", 'r') as file:
        alpha3_to_alpha2 = json.load(file)

    degree = dict(nx.degree_centrality(graph))
    df['iso2'] = df['country_code'].map(alpha3_to_alpha2)
    df['degree'] = df['iso2'].map(degree)

    return df.sort_values(by='degree', ascending=False)[
        'country_code'].iloc[:k].tolist()


def make_train_val_test(df, test_val_prop=0.2):
    """
    Makes a train, validation and test sets according to the desired proportion
    for the test and validation sets. Validation set and test set are the same
    size

    Parameters
    ----------
    df : pandas.DataFrame
    test_val_prop : int, optional

    Returns
    -------
    X_train : pandas.DataFrame
    X_val : pandas.DataFrame
    X_test : pandas.DataFrame
    y_train : pandas.Series
    y_val : pandas.Series
    y_test : pandas.Series

    """

    X = df.drop('total_death', axis=1)
    y = df['total_death']

    train_val_size = int(df.shape[0] * test_val_prop)

    X_train_val, X_test, y_val_train, y_test = train_test_split(
        X, y, test_size=train_val_size, random_state=42, shuffle=True)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_val_train, test_size=train_val_size, random_state=42,
        shuffle=True)
    
    print(f"Train set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

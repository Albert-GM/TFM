# =============================================================================
# Adds features to the data obtained from the simulation to train a machine
# learning model.
# =============================================================================


import pandas as pd
import json
import networkx as nx
import os
import re
import numpy as np

root_project = re.findall(r'(^\S*TFM)', os.getcwd())[0]


def get_data():
    """
    Gets necesseray data for computing the features.

    Returns
    -------
    graph : networkx.graph
    df_info : pandas.DataFrame
    alpha3_to_alpha2 : dictionary

    """

    graph = nx.read_gpickle(
        f'{root_project}/data/interim/routes_countries.gpickle')
    df_info = pd.read_pickle(
        f'{root_project}/data/interim/country_info_final.pickle')

    with open(f'{root_project}/data/interim/alpha3_to_alpha2.txt', 'r') as file:
        alpha3_to_alpha2 = json.load(file)

    return graph, df_info, alpha3_to_alpha2


def feature_graph(df):
    """
    Adds to the dataframe features about the graph that represents the connection
    between countries.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    df : pandas.DataFrame

    """

    data = get_data()
    graph = data[0]
    alpha3_to_alpha2 = data[2]

    degree = dict(nx.degree_centrality(graph))
    betw = nx.betweenness_centrality(graph)
    closeness = nx.closeness_centrality(graph)

    df['iso2'] = df['i_country'].map(alpha3_to_alpha2)
    df['betweenness'] = df['iso2'].map(betw)
    df['degree'] = df['iso2'].map(degree)
    df['closeness'] = df['iso2'].map(closeness)
    df.drop(labels='iso2', axis=1, inplace=True)

    return df


def feature_pop(df):
    """
    Adds to the dataframe the population about the initial country where the
    disease begins.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    df : pandas.DataFrame

    """

    df_info = get_data()[1]
    dict_pop_country = df_info[['country_code', 'total_pop']].set_index(
        'country_code').iloc[:, 0].to_dict()
    df['country_pop'] = df['i_country'].map(dict_pop_country)

    return df


def feature_total_dep(df):
    """
    Adds to the dataframe the population about the initial country where the
    disease begins.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    df : pandas.DataFrame

    """

    df_info = get_data()[1]
    df_info['total_departures'] = df_info['departures/day'].apply(
        lambda x: np.array(list(x.values())).sum())
    dict_total_dep = df_info[['country_code', 'total_departures']].set_index(
        'country_code').iloc[:, 0].to_dict()
    df['country_departures'] = df['i_country'].map(dict_total_dep)

    return df


def feature_exposed_pop(df):
    """
    Adds the total population of the countries to which an individual can travel
    from the initial country. Is the population most exposed to the disease
    apart from the initial country.

    Parameters
    ----------
    df :  pandas.DataFrame

    Returns
    -------
    df : pandas.DataFrame

    """
    df_info = get_data()[1]
    pop_dict = df_info[['country_code', 'total_pop']].set_index(
        'country_code').iloc[:, 0].to_dict()
    df_info['exposed_population'] = df_info['departures/day'].apply(
        lambda x: np.array([pop_dict[country] for country in x.keys()]).sum())
    exposed_dict = df_info[['country_code', 'exposed_population']].set_index(
        'country_code').iloc[:, 0].to_dict()
    df['exposed_pop'] = df['i_country'].map(exposed_dict)

    return df


def feature_transf_log(df):
    """
    Applies a logarithmic transformation to the fatures disesase.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    df : pandas.DataFrame

    """
    # Replace 0 by a infinitesimal number to avoid -infinity
    df['inf_pow_1_log'] = np.log(
        df['inf_pow_1'].replace(
            0, np.finfo(float).eps))
    df['inf_pow_2_log'] = np.log(
        df['inf_pow_2'].replace(
            0, np.finfo(float).eps))
    df['mort_pow_1_log'] = np.log(
        df['mort_pow_1'].replace(
            0, np.finfo(float).eps))
    df['mort_pow_2_log'] = np.log(
        df['mort_pow_2'].replace(
            0, np.finfo(float).eps))
    df['mort_pow_3_log'] = np.log(
        df['mort_pow_3'].replace(
            0, np.finfo(float).eps))

    return df


def add_features(df):
    """
    Adds all the features to the dataframe

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    df : pandas.DataFrame

    """

    df = feature_graph(df)
    df = feature_pop(df)
    df = feature_total_dep(df)
    df = feature_exposed_pop(df)
    df = feature_transf_log(df)
    
    return df

if __name__ == '__main__':
    
    df_v1 = pd.read_csv(
        f"{root_project}/data/processed/simulation_results_v1.csv")
    df_v2 = pd.read_csv(
        f"{root_project}/data/processed/simulation_results_v2.csv")
    
    df_v1 = add_features(df_v1)
    test_size= 120000
    df_test = df_v1.iloc[:test_size]
    df_v1_train_val = df_v1.iloc[test_size:]

    df_v2_train_val = add_features(df_v2)
    
    df_train_val_set = pd.concat([df_v1_train_val,
                                  df_v2_train_val],
                                 ignore_index=True)
    
    df_train_val_set = df_train_val_set.sample(frac=1).reset_index(drop=True)


    print(f"Test size: {df_test.shape[0]}")
    print(f"Train validation size (v1): {df_v1_train_val.shape[0]}")
    print(f"Train validation size (v2): {df_v2_train_val.shape[0]}")

    
    df_test.to_pickle(
        f"{root_project}/data/processed/test_set.pickle")
    
    df_v1_train_val.to_pickle(
        f"{root_project}/data/processed/train_val_set_v1.pickle")
    
    df_v2_train_val.to_pickle(
        f"{root_project}/data/processed/train_val_set_v2.pickle")
    
    df_train_val_set.to_pickle(
        f"{root_project}/data/processed/train_val_set.pickle")
    
    
    df_test.to_csv(
        f"{root_project}/data/processed/test_set.csv", index=False)
    
    df_v1_train_val.to_csv(
        f"{root_project}/data/processed/train_val_set_v1.csv", index=False)
    
    df_v2_train_val.to_csv(
        f"{root_project}/data/processed/train_val_set_v2.csv", index=False)
    
    df_train_val_set.to_csv(
        f"{root_project}/data/processed/train_val_set.csv", index=False)   
    
    
    

# =============================================================================
# Adds features to the data obtained from the simulation to train a machine
# learning model.
# =============================================================================


import pandas as pd
import json
import networkx as nx
import os
import re

root_project = re.findall(r'(^\S*TFM-master)', os.getcwd())[0]


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


def features_graph(df):
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


def features_pop(df):
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





if __name__ == '__main__':
    df_1 = pd.read_csv(f'{root_project}/data/processed/simulation_results.csv')
    df_2 = pd.read_csv(f'{root_project}/data/processed/simulation_results_v2.csv')
    df_3 = pd.read_csv(f'{root_project}/data/processed/simulation_results_v3.csv')
    df_4 = pd.read_csv(f'{root_project}/data/processed/simulation_results_v4_errors.csv')
    df = pd.concat([df_1, df_2, df_3, df_4], ignore_index=True)
    # add new features to the sir simulation results
    df = features_graph(df)
    df = features_pop(df)

    df.to_pickle(f"{root_project}/data/processed/features_model_rev5.pickle")
















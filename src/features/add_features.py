import pandas as pd
import json
import networkx as nx
import os
import re
from sklearn.model_selection import train_test_split

root_project = re.findall(r'(^\S*TFM_AGM)', os.getcwd())[0]


def get_data():

    graph = nx.read_gpickle(
        f'{root_project}/data/interim/routes_countries.gpickle')
    df_info = pd.read_pickle(
        f'{root_project}/data/interim/country_info_final.pickle')

    with open(f'{root_project}/data/interim/alpha3_to_alpha2.txt', 'r') as file:
        alpha3_to_alpha2 = json.load(file)

    return graph, df_info, alpha3_to_alpha2


def features_graph(df):

    data = get_data()
    graph = data[0]
    alpha3_to_alpha2 = data[2]

    degree = dict(nx.degree_centrality(graph))
    betw = nx.betweenness_centrality(graph)
    closeness = nx.closeness_centrality(graph)

    df['iso2'] = df['initial_country'].map(alpha3_to_alpha2)
    df['betweenness'] = df['iso2'].map(betw)
    df['degree'] = df['iso2'].map(degree)
    df['closeness'] = df['iso2'].map(closeness)
    df.drop(labels='iso2', axis=1, inplace=True)

    return df


def features_pop(df):

    df_info = get_data()[1]
    dict_pop_country = df_info[['country_code', 'total_pop']].set_index(
        'country_code').iloc[:, 0].to_dict()
    df['country_pop'] = df['initial_country'].map(dict_pop_country)

    return df


def make_train_val_test(df, test_val_prop=0.2):

    X = df.drop('total_death', axis=1)
    y = df['total_death']

    train_val_size = int(df.shape[0] * test_val_prop)

    X_train_val, X_test, y_val_train, y_test = train_test_split(
        X, y, test_size=train_val_size, random_sate=42, shuffle=True)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_val_train, test_size=train_val_size, random_sate=42,
        shuffle=True)

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == '__main__':
    df_1 = pd.read_pickle(
        f'{root_project}/data/processed/sir_simulation_50k_rev10.pickle')
    df_2 = pd.read_pickle(
        f'{root_project}/data/processed/sir_simulation_50k_v2_rev10.pickle')
    df_3 = pd.read_pickle(
        f'{root_project}/data/processed/sir_simulation_10k_v2_rev10_2.pickle')
    # from here is data based on mistakes of previous models
    df_4 = pd.read_pickle(
        f'{root_project}/data/processed/sir_simulation_50k_v3_rev10.pickle')
    df_main = pd.concat([df_1, df_2, df_3, df_4])
    df_main.reset_index(inplace=True, drop=True)
    # add new features to the sir simulation results
    df_main = features_graph(df_main)
    df_main = features_pop(df_main)

    df_main.to_pickle(f"{root_project}/data/processed/features_model.pickle")

    # divide de data in test, validation and train sets

    # train_size = 0.6
    # test_val_size = (1 - train_size) / 2
    # test_val_size = int(df_main.shape[0] * test_val_size)

    # df_test = df_main.iloc[:test_val_size].sample(
    #     frac=1).reset_index(drop=True)
    # df_val = df_main.iloc[test_val_size:test_val_size *
    #                       2].sample(frac=1).reset_index(drop=True)
    # df_train = df_main.iloc[test_val_size *
    #                         2:].sample(frac=1).reset_index(drop=True)

    # df_test.to_pickle(f"{root_project}/data/processed/test_set.pickle")
    # df_val.to_pickle(f"{root_project}/data/processed/val_set.pickle")
    # df_train.to_pickle(f"{root_project}/data/processed/train_set.pickle")

import pandas as pd
import json
import networkx as nx
import os, re

root_project = re.findall(r'(^\S*TFM_AGM)', os.getcwd())[0]

def get_data():
    
    graph = nx.read_gpickle(f'{root_project}/data/interim/routes_countries.gpickle')
    df_info = pd.read_pickle(f'{root_project}/data/interim/country_info_final.pickle')
    
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


def features_neigh(df):
    
    df_info = get_data()[1]
    print("hola")


    
    
if __name__ == '__main__':
    df_main = pd.read_pickle(f'{root_project}/data/processed/sim_closure_errors_50k.pickle')
    df_main = features_graph(df_main)
    df_main = features_pop(df_main)
    df_main = features_neigh(df_main)
    df_main.to_pickle(f'{root_project}/data/processed/features_model.pickle')

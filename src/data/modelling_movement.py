# =============================================================================
# Makes a model simulating the movement of people between countries.
# =============================================================================


import pandas as pd
import numpy as np
import networkx as nx
import os
import re
root_project = re.findall(r'(^\S*TFM)', os.getcwd())[0]

df = pd.read_pickle(f"{root_project}/data/interim/country_info_nonans.pickle")

df_move = df.loc[:, ['country_name', 'country_code']]
# arrivals and departures by country and day
df_move['arrivals/day'] = df['arrivals'] / 365
df_move['departures/day'] = df['departures'] / 365
# ratio of arrivals to total by country
df_move['prop_arrivals'] = df_move['arrivals/day'] / \
    np.sum(df_move['arrivals/day'])

# probability of arrival by country
countrycode_to_proparriv = pd.Series(
    df_move['prop_arrivals'].values, index=df_move['country_code']).to_dict()
# departures by country
countrycode_to_departures = pd.Series(
    df_move['departures/day'].values, index=df_move['country_code']).to_dict()


# adds to the dataframe a column with info about the number of people going from
# one country to another
l_people = []
df_people = df.copy()

for country in df.iterrows():
    # possibles destinations of the country
    country_destinations = country[1]['destinations']
    # probability of arrival by posible destination
    prob = {x: countrycode_to_proparriv[x] for x in country_destinations}
    sum_prob = np.sum(list(prob.values()))
    # probability of going to each of destinations normalized. sum=1
    prob = {k: v / sum_prob for k, v in prob.items()}
    # compute number of individuals going from country to possible destinations
    people = {k: int(round(
        v * countrycode_to_departures[country[1]['country_code']], 0))
        for k, v in prob.items()}
    l_people.append(people)


df['departures/day'] = l_people
# drops destinations columns because more information is in departures/day
df.drop('destinations', axis=1, inplace=True)

# makes an origin-destination matrix from the dataframe
H = nx.DiGraph()

for index, country in df.iterrows():
    destinations = country['departures/day']
    for k, v in destinations.items():
        H.add_edge(country['country_code'], k, people=v)

OD_matrix = nx.attr_matrix(H, edge_attr='people', rc_order=df['country_code'])

# uncomment to save new data
# df.to_pickle(f"{root_project}/data/interim/country_info_final.pickle")
# np.save(f"{root_project}/data/interim/od_matrix.npy", OD_matrix)

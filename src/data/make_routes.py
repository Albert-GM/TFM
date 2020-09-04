# =============================================================================
# Makes graphs with routes between countries and airports around the world.
# =============================================================================


import pandas as pd
import json
import networkx as nx
import os
import re
root_project = re.findall(r'(^\S*TFM)', os.getcwd())[0]

# Read necessary data
df_routes = pd.read_csv(f"{root_project}/data/raw/routes.csv")
df_iata = pd.read_csv(f"{root_project}/data/raw/airport_codes.csv")
df_iso = pd.read_csv(f"{root_project}/data/raw/tableconvert_iso.csv")
with open(f"{root_project}/data/interim/iata_to_country.txt", 'r') as f:
    iata_to_country = json.load(f)

df_routes.rename(columns={' source airport': 'source_airport',
                          ' destination apirport': 'destination_airport'},
                 inplace=True)

df_routes = df_routes[['source_airport', 'destination_airport']]


df_routes['source_country'] = df_routes['source_airport'].map(iata_to_country)
df_routes['destination_country'] = df_routes['destination_airport'].map(
    iata_to_country)

# Drop rows without destination or source country
df_routes.dropna(inplace=True)
# Drop rows where source == destiniation
df_routes = df_routes.loc[df_routes['source_country'] !=
                          df_routes['destination_country'], :]

# Make graphs from dataframe
G_country = nx.from_pandas_edgelist(
    df_routes,
    'source_country',
    'destination_country',
    create_using=nx.DiGraph())

G_airport = nx.from_pandas_edgelist(
    df_routes,
    'source_airport',
    'destination_airport',
    create_using=nx.DiGraph())

# Drop country if country is not in df_iso
H_country = G_country.copy()
for country in G_country:
    if country not in df_iso['Alpha-2 code'].unique():
        H_country.remove_node(country)

# Add airport information to airport graph
df_iata.drop_duplicates(subset='iata_code', inplace=True)

H_airport = G_airport.copy()

for airport in H_airport.nodes:
    H_airport.nodes[airport]['name'] = df_iata.loc[df_iata['iata_code']
                                                   == airport, 'name'].item()
    H_airport.nodes[airport]['iso_country'] = df_iata.loc[df_iata['iata_code']
                                                          == airport, 'iso_country'].item()
    H_airport.nodes[airport]['coordinates'] = df_iata.loc[df_iata['iata_code']
                                                          == airport, 'coordinates'].item()
    H_airport.nodes[airport]['continent'] = df_iata.loc[df_iata['iata_code']
                                                        == airport, 'continent'].item()


nx.write_gpickle(
    H_country,
    f"{root_project}/data/interim//routes_countries.gpickle")
nx.write_gpickle(
    G_airport,
    f"{root_project}/data/interim/routes_airports.gpickle")

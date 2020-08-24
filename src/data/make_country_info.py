# =============================================================================
# Makes a dataframe with all the necessary data about countries.
# =============================================================================


# Add project directory to pythonpath to import own functions
import sys, os ,re
root_project = re.findall(r'(^\S*TFM-master)', os.getcwd())[0]
sys.path.append(root_project)

from src.utils.help_func import extract_indicator, last_values
import json
import networkx as nx
import numpy as np
import pandas as pd



# Read necessary data
df_indicators = pd.read_csv(
    f"{root_project}/data/raw/world_indicators_data.csv")
df_pop_plus = pd.read_csv(
    f"{root_project}/data/raw/country_population.csv",
    skiprows=4)
df_location = pd.read_csv(f"{root_project}/data/raw/tableconvert_iso.csv")
df_continents = pd.read_csv(
    f"{root_project}/data/raw/country_to_continent.csv")
G = nx.read_gpickle(f"{root_project}/data/interim/routes_countries.gpickle")
with open(f"{root_project}/data/interim/alpha2_to_alpha3.txt", 'r') as file:
    alpha2_to_alpha3 = json.load(file)

# Extract data about population
df_male = last_values(extract_indicator(df_indicators, 'Population, male'))
df_male.rename(
    columns={
        'Country Name': 'country_name',
        'Country Code': 'country_code',
        'last_value': 'male_pop'},
    inplace=True)
df_female = last_values(extract_indicator(df_indicators, 'Population, female'))
df_female.rename(columns={'last_value': 'female_pop'}, inplace=True)

df_population = pd.concat([df_male, df_female.loc[:, 'female_pop']], axis=1)
df_population['total_pop'] = df_population['male_pop'] + \
    df_population['female_pop']
df_population = df_population[['country_name', 'country_code', 'total_pop']]
# Add second source to fill nans
dict_pop = df_pop_plus[['Country Code', '2019']].set_index(
    'Country Code', drop=True).iloc[:, 0].to_dict()
df_population['total_pop'] = df_population['total_pop'].fillna(
    df_population['country_code'].map(dict_pop))

# Extract data about arrivals
df_arrivals = extract_indicator(df_indicators,
                                'International tourism, number of arrivals')
df_arrivals = last_values(df_arrivals)
df_arrivals.rename(
    columns={
        'Country Name': 'country_name',
        'Country Code': 'country_code',
        'last_value': 'arrivals'},
    inplace=True)

# Extract data about departures
df_departures = extract_indicator(
    df_indicators,
    'International tourism, number of departures')
df_departures = last_values(df_departures)
df_departures.rename(
    columns={
        'Country Name': 'country_name',
        'Country Code': 'country_code',
        'last_value': 'departures'},
    inplace=True)

# Merge dataframes
df_full = pd.merge(df_population, df_arrivals)
df_full = pd.merge(df_full, df_departures)
df_location.rename(
    columns={
        'Alpha-3 code': 'country_code',
        'Latitude (average)': 'latitude',
        'Longitude (average)': 'longitude'},
    inplace=True)
# Drop duplicates in df_location and merge
df_full = pd.merge(df_full,
                   df_location.loc[:,
                                   ['country_code',
                                    'latitude',
                                    'longitude']].drop_duplicates(),
                   on='country_code',
                   how='left')


# If latitude is nan, it is not a country
df_full.dropna(subset=['latitude'], inplace=True)
df_full.reset_index(drop=True, inplace=True)

# Add ratios
df_full['arrivals/total'] = df_full['arrivals'] / np.sum(df_full['arrivals'])
df_full['departures/total'] = df_full['departures'] / \
    np.sum(df_full['departures'])
df_full['arrivals/population'] = df_full['arrivals'] / df_full['total_pop']
df_full['departures/population'] = df_full['departures'] / df_full['total_pop']

# Add continent
df_continents.rename(
    columns={
        'Three_Letter_Country_Code': 'country_code',
        'Continent_Name': 'continent_name',
        'Continent_Code': 'continent_code'},
    inplace=True)
df_continents = df_continents.loc[:, [
    'continent_name', 'continent_code', 'country_code']]
df_full = pd.merge(df_full, df_continents, how='left')
df_full.drop_duplicates(subset=['country_code'], inplace=True)

# Dictionary with destinations by country, if country is not in df_full drop it
country_to_destination = {}  # dictionary with possible desinations of countries

for country in G:
    destinations = list(G[country])
    destinations_rev = destinations.copy()
    if alpha2_to_alpha3[country] not in df_full['country_code'].values:
        continue
    if country in destinations:  # a country cannot be a destination from himself
        destinations_rev.remove(country)
    for dest in destinations:
        if alpha2_to_alpha3[dest] not in df_full['country_code'].values:
            destinations_rev.remove(dest)
    country_to_destination[country] = destinations_rev

# Map country-destination from ISO 2 to ISO 3
country_to_destination = {
    alpha2_to_alpha3[k]: [
        alpha2_to_alpha3[x] for x in v] for k,
    v in country_to_destination.items()}
df_full['destinations'] = df_full['country_code'].map(country_to_destination)
# Drop rows where country has no possible destinations
df_full.dropna(subset=['destinations'], inplace=True)
df_full.reset_index(drop=True, inplace=True)

df_full.to_pickle(f"{root_project}/data/interim/country_info.pickle")

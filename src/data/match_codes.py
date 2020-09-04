# =============================================================================
# Makes dictionaries mapping ISO codes of countrys and IATA codes to ISO codes.
# =============================================================================

import pandas as pd
import json
import os
import re
root_project = re.findall(r'(^\S*TFM)', os.getcwd())[0]

# Read necessary data
df_iso = pd.read_csv(f"{root_project}/data/raw/tableconvert_iso.csv")
df_iata = pd.read_csv(f"{root_project}/data/raw/airport_codes.csv")

# Make ISO mapping dictionaries
alpha2_to_alpha3 = pd.Series(df_iso['Alpha-3 code'].values,
                             index = df_iso['Alpha-2 code']).to_dict()

alpha3_to_alpha2 = pd.Series(df_iso['Alpha-2 code'].values,
                             index = df_iso['Alpha-3 code']).to_dict()

alpha2_to_country = pd.Series(df_iso['Country'].values,
                              index = df_iso['Alpha-2 code']).to_dict()

alpha3_to_country = pd.Series(df_iso['Country'].values,
                              index = df_iso['Alpha-3 code']).to_dict()

# Make IATA mapping, include only open airports and not nan
df_iata = df_iata.loc[(df_iata['type'] != 'closed') &
                        (~df_iata['iata_code'].isna()) &
                        (~df_iata['iso_country'].isna()),
              ['name', 'iso_country', 'iata_code', 'continent', 'coordinates']]

iata_to_country = pd.Series(df_iata['iso_country'].values,
                            index = df_iata['iata_code']).to_dict()



with open(f"{root_project}/data/interim/alpha2_to_alpha3.txt", 'w') as file:
    json.dump(alpha2_to_alpha3, file)
    
with open(f"{root_project}/data/interim/alpha3_to_alpha2.txt", 'w') as file:
    json.dump(alpha3_to_alpha2, file)
    
with open(f"{root_project}/data/interim/alpha2_to_country.txt", 'w') as file:
    json.dump(alpha2_to_country, file)
    
with open(f"{root_project}/data/interim/alpha3_to_country.txt", 'w') as file:
    json.dump(alpha3_to_country, file)
    
with open(f"{root_project}/data/interim/iata_to_country.txt", 'w') as file:
    json.dump(iata_to_country, file)
    
    
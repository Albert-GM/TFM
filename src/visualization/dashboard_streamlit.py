# Add project directory to pythonpath to import own functions
import sys, os ,re
root_project = re.findall(r'(^\S*TFM)', os.getcwd())[0]
sys.path.append(root_project)

from src.features.sird_model import SIRD_model
import pandas as pd
import plotly.express as px 
import numpy as np
import streamlit as st


df = pd.read_pickle(f"{root_project}/data/interim/country_info_final.pickle")
# df_pop = pd.read_pickle(
# f"{root_project}/data/interim/country_info_final.pickle")

d_mapping = df[['country_name','country_code']].set_index(
    'country_name').iloc[:,0].to_dict()

max_width = 3000
padding_top = 0
padding_right =0
padding_left = 5
padding_bottom = 0
COLOR = 'black'
BACKGROUND_COLOR =  'white'

st.markdown(
        f"""
<style>
    .reportview-container .main .block-container{{
        max-width: {max_width}px;
        padding-top: {padding_top}rem;
        padding-right: {padding_right}rem;
        padding-left: {padding_left}rem;
        padding-bottom: {padding_bottom}rem;
    }}
    .reportview-container .main {{
        color: {COLOR};
        background-color: {BACKGROUND_COLOR};
    }}
</style>
""",
        unsafe_allow_html=True,
    )


st.write("""
# SPREAD OF A PANDEMIC WORLDWIDE

This application shows the spread of an infectious disease around the world.

""")



st.sidebar.header('User Input Parameters')



l_countries = ['Spain',
                'France',
                'United States',
                'Argentina',
                'Germany',
                'United Kingdom',
                'China',
                'Cameroon',
                'Chile',
                'Italy',
                'Mexico',
                'Zambia',
                'Jamaica',
                'Haiti',
                'Pakistan',
                'Morocco',
                'Vietnam',
                'India',
                'Thailand']

l_countries.sort()
idx = l_countries.index('Spain')

def user_input_features():
    R0 = st.sidebar.slider('Reproduction number', 2, 18, 4, step=1)
    Tr = st.sidebar.slider('Recovery time (s)', 2, 30, 20, step=1)
    omega = st.sidebar.slider('Fatility rate (%)', 1, 90, 2, step=2)
    reaction = st.sidebar.slider('Reaction time (days)', 1, 20, 1, step=1)
    n_closed = st.sidebar.slider('Countries in quarantine', 0, 18, 0, step=1)
    i_country = st.sidebar.selectbox('Initial country', l_countries, idx)
    
    param_grid = {'R0' : R0,
                  'Tr' : Tr,
                  'omega' : omega/100,
                  'react_time' : reaction,
                  'n_closed' : n_closed,
                  'countries' : i_country }    
    
    return param_grid

parameters = user_input_features()

parameters['countries'] = d_mapping[parameters['countries']]

# st.write(parameters)

# Features to keep
dict_keys = [
    'SIRD_t',
    'SIRD_p_t',
    'SIRD_world_p_t',
    'SIRD_world_t',
    'total_infected',
    'total_deceased'

]
    
sir_model = SIRD_model(
    parameters['R0'],
    parameters['Tr'],
    parameters['omega'],
    parameters['countries'],
    parameters['n_closed'],
    parameters['react_time'])
sir_model.simulate()
sir_model.compute_disease_features()
data = sir_model.get_simulation_data() 
subset_data = {column: data[column] for column in dict_keys}
    
s = pd.Series(subset_data)

value = s['SIRD_world_t'][1,:].max()*0.0001
zoom = pd.Series(s['SIRD_world_t'][1,:])
mask = zoom[zoom>value].index


data = pd.DataFrame(s['SIRD_t'][:,1,:]).T
data = data.iloc[mask].T
data = data.groupby(np.arange(len(data.columns))//7, axis=1).mean()
data['country_code'] = df['country_code']


data_melted = pd.melt(frame=data, id_vars='country_code', var_name='week',
                      value_name='infected' )
data_melted['color'] = 'red'
# Plotly Express
fig_world = px.scatter_geo(data_melted, locations="country_code",
                      hover_name="country_code", size="infected",
                      animation_frame='week', 
                      size_max=30, color='color',
                      color_discrete_map={'red':'rgb(255,0,0)'},
                      width=800, height=600)

fig_world.update_layout(showlegend=False) 


frame = {'susceptible': s['SIRD_world_p_t'][0, :],
        'infected':s['SIRD_world_p_t'][1, :],
        'recovered':s['SIRD_world_p_t'][2, :],
        'deceased':s['SIRD_world_p_t'][3, :]}   

df_sird = pd.DataFrame(frame)
df_sird = df_sird.loc[mask].T
df_sird = df_sird.groupby(np.arange(len(df_sird.columns))//7, axis=1).mean().T
df_sird.reset_index(inplace=True)
df_sird = pd.melt(df_sird, id_vars='index',
              value_vars=['susceptible', 'infected','recovered','deceased'])
df_sird.rename(columns={'index': 'weeks', 'value': 'proportion',
                        'variable': 'compartment'}, inplace=True)
fig_sird = px.line(df_sird , x="weeks", y="proportion", color='compartment',
                    width=800, height=400)
fig_sird.update_layout(legend_title_text='')


N = df['total_pop'].sum()

prop_infected = (s['total_infected'] / N) *100
prop_deceased = (s['total_deceased'] / N) * 100

string_white = "&nbsp;"*40
st.markdown(f"""
## {string_white}**{prop_infected:.2f} % infected**
## {string_white}**{prop_deceased:.2f} % deceased**
""")

st.write("Representation of SIRD model:")

st.write(fig_sird)

st.write("""
World map showing the spread of the disease in various countries over the weeks:
""")


st.write(fig_world)

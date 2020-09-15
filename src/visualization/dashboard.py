# Add project directory to pythonpath to import own functions
import sys, os ,re
root_project = re.findall(r'(^\S*TFM)', os.getcwd())[0]
sys.path.append(root_project)


import pandas as pd
import plotly.express as px 
import numpy as np
import dash  
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from src.features.sird_model import SIRD_model


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

df = pd.read_pickle(f"{root_project}/data/interim/country_info_nonans.pickle")



# App layout
app.layout = html.Div([

    html.Div([
    html.H1("Spread of a pandemic", style={'text-align': 'center'})]),
    
    html.Div([
    html.Label("Select the reproduction number"),
    dcc.Dropdown(id="select_R0",
                  options=[
                      {"label": "2", "value": 2},
                      {"label": "4", "value": 4},
                      {"label": "6", "value": 6},
                      {"label": "8", "value": 8}],
                  multi=False,
                  value=4, 
                  ),

    html.Label("Select the recovery time"),
    dcc.Dropdown(id="select_Tr",
                  options=[
                      {"label": "5", "value": 5},
                      {"label": "10", "value": 10},
                      {"label": "20", "value": 20}],
                  multi=False,
                  value=10,
                  ),
    html.Label("Select the mortality rate"),
    dcc.Dropdown(id="select_omega",
                  options=[
                      {"label": "0.01", "value": 0.01},
                      {"label": "0.05", "value": 0.05},
                      {"label": "0.1", "value": 0.1},
                      {"label": "0.2", "value": 0.2}
                      ],
                  multi=False,
                  value=0.01, 
                  ),
    
    html.Label("Select the initial country"),
    dcc.Dropdown(id="select_country",
                  options=[
                      {"label": "Spain", "value": 'ESP'},
                      {"label": "France", "value": 'FRA'},
                      {"label": "China", "value": 'CHN'},
                      {"label": "Italy", "value": 'ITA'},
                      {"label": "United States", "value": 'USA'}
                      ],
                  multi=False,
                  value='ESP', 
                  ),
    
    html.Label("Select number of countries to close"),
    dcc.Dropdown(id="select_close",
                  options=[
                      {"label": "0", "value": 0},
                      {"label": "1", "value": 1},
                      {"label": "5", "value": 5},
                      {"label": "10", "value": 10},
                      {"label": "20", "value": 20}
                      ],
                  multi=False,
                  value=0, 
                  ),
    
    html.Label("Select the reaction time in days"),
    dcc.Dropdown(id="select_reaction",
                  options=[
                      {"label": "1", "value": 1},
                      {"label": "5", "value": 5},
                      {"label": "10", "value": 10},
                      {"label": "30", "value": 30}
                      ],
                  multi=False,
                  value=5, 
                  )]),
    html.Div([
    dcc.Graph(id='world_map', figure={})
    ], style = {'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}),
    
    html.Div([
    dcc.Graph(id='sird_model', figure={})
    ], style = {'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'})    
    
    
    
])



# Connect the Plotly graphs with Dash Components
@app.callback(
    [Output(component_id='world_map', component_property='figure'),
     Output(component_id='sird_model', component_property='figure')],
    [Input(component_id='select_R0', component_property='value'),
     Input(component_id='select_Tr', component_property='value'),
     Input(component_id='select_omega', component_property='value'),
     Input(component_id='select_country', component_property='value'),
     Input(component_id='select_close', component_property='value'),
     Input(component_id='select_reaction', component_property='value')
     ]
)
def update_graph(R0, Tr, omega, country, close, reaction): # as many arguments as inputs
    print([R0, Tr, omega, country, close, reaction])




    # Features to keep
    dict_keys = [
        'SIRD_t',
        'SIRD_p_t',
        'SIRD_world_p_t',
        'SIRD_world_t'
    ]
    
    
    param = {'R0': R0,
      'Tr': Tr,
      'countries': country,
      'n_closed': close,
      'omega': omega,
      'react_time': reaction}
    

    sir_model = SIRD_model(
        param['R0'],
        param['Tr'],
        param['omega'],
        param['countries'],
        param['n_closed'],
        param['react_time'])
    sir_model.simulate()
    sir_model.compute_disease_features()
    data = sir_model.get_simulation_data() # Get the data in a dict
    subset_data = {column: data[column] for column in dict_keys}
        
    s = pd.Series(subset_data)
    
    value = s['SIRD_world_t'][1,:].max()*0.0001
    zoom = pd.Series(s['SIRD_world_t'][1,:])
    mask = zoom[zoom>value].index
    
    
    data = pd.DataFrame(s['SIRD_t'][:,1,:]).T
    data = data.iloc[mask].T
    data = data.groupby(np.arange(len(data.columns))//7, axis=1).mean()
    print(data)
    data['country_code'] = df['country_code']
    
    
    data_melted = pd.melt(frame=data, id_vars='country_code', var_name='week',
                          value_name='infected' )
    data_melted['color'] = 'red'
    # Plotly Express
    fig_world = px.scatter_geo(data_melted, locations="country_code",
                          hover_name="country_code", size="infected",
                          animation_frame='week', 
                          size_max=60, color='color',
                          color_discrete_map={'red':'rgb(255,0,0)'},
                          width=2500, height=1000)
    
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
                       width=1300, height=600)
    fig_sird.update_layout(legend_title_text='')

    return fig_world, fig_sird # as many arguments as ouptus, in the same order





if __name__ == '__main__':
    app.run_server(debug=True)
    
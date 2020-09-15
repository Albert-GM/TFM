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
from src.features.sird_model_simple import SIRD_model_simple


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)



# App layout
app.layout = html.Div([

    html.Div([
    html.H1("SIRD model", style={'text-align': 'center'})]),
    
    html.Div([
    html.Label("Select the population"),
    dcc.Input(id="select_N", type="number", debounce=True, placeholder="Population", value=10000),

    html.Label("Select the initial infected"),
    dcc.Input(id="select_I0", type="number",debounce=True, placeholder="Initial infected", value=1),

    html.Label("Select the reproduction number"),
    dcc.Input(id="select_R0", type="number",debounce=True, placeholder="Reproduction number", value=4),
    
    html.Label("Select the recovery time"),
    dcc.Input(id="select_Tr", type="number",debounce=True, placeholder="Recovery time", value=20),

    html.Label("Select the moratlity rate"),
    dcc.Input(id="select_omega", type="number",debounce=True, placeholder="Mortality rate", value=0.01),
 
    html.Label("Select the simulation time"),
    dcc.Input(id="select_T", type="number", debounce=True,placeholder="Simulation time", value=200),
    ],style={'float': 'left','margin': 'auto'}),
    
    html.Div([
    dcc.Graph(id='sird_model', figure={})
    ],style={'float': 'right','margin': 'auto'}),
    
    
])
                
                
# Connect the Plotly graphs with Dash Components
@app.callback(
    Output(component_id='sird_model', component_property='figure'),
    [Input(component_id='select_N', component_property='value'),
     Input(component_id='select_I0', component_property='value'),
     Input(component_id='select_R0', component_property='value'),
     Input(component_id='select_Tr', component_property='value'),
     Input(component_id='select_omega', component_property='value'),
     Input(component_id='select_T', component_property='value')
     ]
)



def update_graph(N, I0, R0, Tr, omega, T): # as many arguments as inputs
    print([N, I0, R0, Tr, omega, T])

    S, I, R, D = SIRD_model_simple(N, I0, R0, Tr, omega, T)

    frame = {'susceptible': S,
        'infected':I,
        'recovered':R,
        'deceased':D}   

    df_sird = pd.DataFrame(frame)
    df_sird.reset_index(inplace=True)        
    df_sird = pd.melt(df_sird, id_vars='index',
                 value_vars=['susceptible', 'infected','recovered','deceased'])
    df_sird.rename(columns={'index': 'days', 'value': 'individuals',
                            'variable': 'compartment'}, inplace=True)
    
    fig = px.line(df_sird , x="days", y="individuals", color='compartment',
                       width=1300, height=500)
    fig.update_layout(legend_title_text='')

    return fig





if __name__ == '__main__':
    app.run_server(debug=True)




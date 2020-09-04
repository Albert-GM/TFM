# allows to import own functions
import sys
import os
import re
root_project = re.findall(r'(^\S*TFM)', os.getcwd())[0]
sys.path.append(root_project)

from src.utils.help_func import results_searchcv,plot_predictions,\
    errors_distribution, plot_visualizations, get_model_data
from sklearn.model_selection import GridSearchCV, train_test_split
import pandas as pd
from scipy.stats import randint
import joblib
import seaborn as sns
sns.set()
import time
from scipy.stats import uniform, randint, loguniform
from sklearn.dummy import DummyRegressor


# Get the data
df_train_val = get_model_data(500000)

# Feature selection
features = [
    'Tr',
    'inf_pow_1',
    'inf_pow_2',
    'mort_pow_1',
    'mort_pow_2',
    'mort_pow_3',
    'n_closed',
    'react_time',
    'total_deceased',
    'betweenness',
    'degree',
    'closeness',
    'country_pop',
    'country_departures',
    'exposed_pop',
    'inf_pow_1_log',
    'inf_pow_2_log',
    'mort_pow_1_log',
    'mort_pow_2_log',
    'mort_pow_3_log',
    ]

df_train_val = df_train_val[features]

print("=" * 20)
print(f"Train_validation size: {df_train_val.shape}")
print("=" * 20)


X_train_val = df_train_val.drop('total_deceased', axis=1)
y_train_val = df_train_val['total_deceased']
X_train, X_val, y_train, y_val = train_test_split(X_train_val,
                                                  y_train_val,
                                                  random_state=42)



param_grid = dict(
    strategy=['mean', 'median'])

scoring = {'R2': 'r2', 'RMSE': 'neg_root_mean_squared_error',
           'MAE': 'neg_mean_absolute_error'}

grid_search = GridSearchCV(DummyRegressor(),
                            param_grid=param_grid,
                            scoring=scoring,
                            refit='R2',                            
                           verbose=1, n_jobs=-1)

grid_search.fit(X_train_val, y_train_val)


# Train the model with only train data and best parameters of random search
estimator = DummyRegressor(**grid_search.best_params_)
estimator.fit(X_train, y_train)

results_searchcv(grid_search, estimator, X_val, y_val)
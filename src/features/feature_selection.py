# allows to import own functions
import sys
import os
import re
root_project = re.findall(r'(^\S*TFM)', os.getcwd())[0]
sys.path.append(root_project)

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from src.utils.help_func import get_model_data
from sklearn.feature_selection import RFECV
# import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import uniform, expon, randint, truncexpon, loguniform
import numpy as np


df_train_val = get_model_data(n_samples=5000, ratio=0)


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


X_train_val = df_train_val.drop('total_deceased', axis=1)
y_train_val = df_train_val['total_deceased']


# XGBoost
# param_dist = dict(
#     estimator__n_estimators=randint(low=15, high=30),
#     estimator__max_depth=randint(low=5, high=20),
#     estimator__learning_rate=loguniform(0.01, 1),
#     estimator__subsample=uniform(loc=0.6, scale=1-0.8),
#     estimator__colsample_bytree=uniform(loc=0.6, scale=1-0.8),
#     estimator__gamma=[0, 1, 2]
# )
# estimator = xgb.XGBRegressor(random_state=42)


# Random Forest
pipe = Pipeline([
    ('imputer', SimpleImputer()),
    ('estimator', RandomForestRegressor(random_state=42))
])

max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)
param_dist = dict(
    estimator__estimator__n_estimators=randint(low=1, high=1000),
    estimator__estimator__max_features=['auto', 'sqrt'],
    estimator__estimator__max_depth=max_depth,
    estimator__estimator__min_samples_split=randint(low=2, high=11),
    estimator__estimator__min_samples_leaf=randint(low=1, high=5),
    estimator__estimator__bootstrap=[True, False]
)
estimator = pipe


selector = RFECV(estimator, step=1, cv=3, scoring='r2', n_jobs=-1)

grid_search = RandomizedSearchCV(selector, param_distributions=param_dist,
                                 cv=3, verbose=1,
                                 n_iter=10, random_state=42)

grid_search.fit(X_train_val, y_train_val)


print(grid_search.best_estimator_.n_features_)
print(grid_search.best_estimator_.support_)
print(grid_search.best_estimator_.ranking_)
print(grid_search.best_estimator_.grid_scores_)

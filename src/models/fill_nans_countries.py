# =============================================================================
# Fills nans in the country dataframe and creates a new dataframe.
# The objective is not to create a model as accurate as possible to fill in the
# nans, but simply to use a methodology more precise than assigning some kind of
# statistic
# =============================================================================


# Add project directory to pythonpath to import own functions
import sys, os ,re
root_project = re.findall(r'(^\S*TFM-master)', os.getcwd())[0]
sys.path.append(root_project)

import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from src.utils.help_func import results_searchcv


df_countries = pd.read_pickle(f"{root_project}/data/interim/country_info.pickle")

# Fill departures
df_model = df_countries[
    [
        'total_pop',
        'arrivals',
        'departures',
        'longitude',
        'latitude',
        'arrivals/total',
        'arrivals/population',
    ]
]

df_model_train_test = df_model.loc[~df_model['departures'].isna()]
df_model_predict = df_model.loc[df_model['departures'].isna()]

X_train_test = df_model_train_test.drop('departures', axis=1)
y_train_test = df_model_train_test['departures']

X_train, X_test, y_train, y_test = train_test_split(
    X_train_test, y_train_test, random_state=42)


# Train the model
rnd_reg = Pipeline(
    [
        ('preprocess', SimpleImputer()),
        ('regressor', RandomForestRegressor(random_state=42)),
    ]
)

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {
    'preprocess__strategy': ['median', 'mean'],
    'regressor__n_estimators': n_estimators,
    'regressor__max_features': max_features,
    'regressor__max_depth': max_depth,
    'regressor__min_samples_split': min_samples_split,
    'regressor__min_samples_leaf': min_samples_leaf,
    'regressor__bootstrap': bootstrap,
}

# Uncomment next lines for training the model

# randomsearch = RandomizedSearchCV(
#     rnd_reg,
#     random_grid,
#     random_state=42,
#     n_iter=500,
#     n_jobs=-1,
#     verbose=2)
# randomsearch.fit(X_train_test, y_train_test)
# joblib.dump(randomsearch, f"{root_project}/models/randomsearch_departures.pkl")

randomsearch = joblib.load(f"{root_project}/models/randomsearch_departures.pkl")

results_searchcv(randomsearch, X_test, y_test)

# Gridsearch based on the results of randomsearch
params = {
    'preprocess__strategy': ['median'],
    'regressor__n_estimators': [150, 200, 250],
    'regressor__max_features': ['sqrt'],
    'regressor__max_depth': [20, 25, 30, 40],
    'regressor__min_samples_split': [10],
    'regressor__min_samples_leaf': [1, 2, 4],
    'regressor__bootstrap': [True],
}

# Uncomment next lines for training the model

# gridsearch = GridSearchCV(rnd_reg, param_grid=params, n_jobs=1, verbose=2)
# gridsearch.fit(X_train_test, y_train_test)
# joblib.dump(gridsearch, f"{root_project}/models/gridsearch_departures.pkl")

gridsearch = joblib.load(f"{root_project}/models/gridsearch_departures.pkl")


results_searchcv(gridsearch, X_test, y_test)


# Fill nans in departures with predicted values
X_predict = df_model_predict.drop('departures', axis=1)
df_model_predict['predicted_departures'] = gridsearch.predict(X_predict)
df_countries['departures'].fillna(
    value=df_model_predict['predicted_departures'],
    inplace=True)
df_countries['departures/total'] = df_countries['departures'] / \
    df_countries['departures'].sum()
df_countries['departures/population'] = df_countries['departures'] / \
    df_countries['total_pop']


# Train a model to fill nans in arrivals, add the new predicted values from
# the previous step
df_model = df_countries[
    [
        'total_pop',
        'arrivals',
        'latitude',
        'longitude',
        'departures',
        'departures/total',
        'departures/population',
    ]
]

df_model_train_test = df_model.loc[~df_model['arrivals'].isna()]
df_model_predict = df_model.loc[df_model['arrivals'].isna()]

X_train_test = df_model_train_test.drop('arrivals', axis=1)
y_train_test = df_model_train_test['arrivals']

X_train, X_test, y_train, y_test = train_test_split(
    X_train_test, y_train_test, random_state=42)

# Uncomment next lines for training the model

# randomsearch = RandomizedSearchCV(
#     rnd_reg,
#     random_grid,
#     random_state=42,
#     n_iter=500,
#     n_jobs=-1,
#     verbose=2)
# randomsearch.fit(X_train_test, y_train_test)
# joblib.dump(randomsearch, f"{root_project}/models/randomsearch_arrivals.pkl")

randomsearch = joblib.load(f"{root_project}/models/randomsearch_arrivals.pkl")

results_searchcv(randomsearch, X_test, y_test)

# Fill nans in arrivals with predicted values
X_predict = df_model_predict.drop('arrivals', axis=1)
df_model_predict['predicted_arrivals'] = gridsearch.predict(X_predict)
df_countries['arrivals'].fillna(
    value=df_model_predict['predicted_arrivals'],
    inplace=True)
df_countries['arrivals/total'] = df_countries['arrivals'] / \
    df_countries['arrivals'].sum()
df_countries['arrivals/population'] = df_countries['arrivals'] / \
    df_countries['total_pop']
    

df_countries.to_pickle(f"{root_project}/data/interim/country_info_nonans.pickle")


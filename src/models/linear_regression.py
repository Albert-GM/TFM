# =============================================================================
# 
# =============================================================================


# allows to import own functions
import sys
import os
import re
root_project = re.findall(r'(^\S*TFM-master)', os.getcwd())[0]
sys.path.append(root_project)

from scipy.stats import  loguniform
import pandas as pd
from sklearn.model_selection import  RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from src.features.add_features import features_graph, features_pop
from src.utils.help_func import results_searchcv, make_train_val_test
from sklearn.pipeline import Pipeline
from yellowbrick.model_selection import LearningCurve, FeatureImportances
from yellowbrick.regressor import ResidualsPlot
import joblib
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()

# Path to save the model
PATH =  f"{root_project}/models/linear_regression_rev17.pkl"

# Read data
df = pd.read_csv(
    f'{root_project}/data/processed/simulation_results_rev17_wide.csv')
# Load features
df = features_graph(df)
df = features_pop(df)

# size_data = 20000 # enter  desired subset of data
# df = df.sample(size_data)

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
    'country_pop']

# features = [
#     'Tr',
#     'inf_pow_1',
#     'inf_pow_2',
#     'mort_pow_2',
#     'mort_pow_3',
#     'n_closed',
#     'react_time',
#     'total_deceased',
#     'country_pop']

df = df[features]

X_train_val, y_train_val, X_test, y_test = make_train_val_test(df, out_mode=1)

pipe = Pipeline([
    ('preprocess', StandardScaler()),
    ('estimator', SGDRegressor())
])

param_dist = dict(
    estimator__loss = ['squared_loss', 'huber'],
    estimator__penalty=['l2', 'l1'],
    estimator__alpha = loguniform(0.00001, 1)
)


random_search = RandomizedSearchCV(pipe, param_distributions=param_dist,
                                   verbose=1, n_iter=100, 
                                   random_state=42, n_jobs=-1)

random_search.fit(X_train_val, y_train_val)
joblib.dump(random_search, PATH)

# Load a model in path
random_search = joblib.load(PATH)

# Prints out useful information about the model
results_searchcv(random_search, X_test, y_test)

# Plot learning curves
fig, ax = plt.subplots(1, 1, figsize = (10,5))
visualizer = LearningCurve(random_search.best_estimator_, scoring='r2')
visualizer.fit(X_train_val, y_train_val)        # Fit the data to the visualizer
visualizer.show()           # Finalize and render the figure


# Plot feature importances
fig, ax = plt.subplots(1, 1, figsize = (10,5))
viz = FeatureImportances(random_search.best_estimator_['estimator'], relative=False)
viz.fit(X_train_val, y_train_val)
viz.show()

# Plot residual plots
fig, ax = plt.subplots(1, 1, figsize = (10,5))
viz = ResidualsPlot(random_search.best_estimator_)
viz.fit(X_train_val, y_train_val)
viz.score(X_test, y_test)
viz.show()
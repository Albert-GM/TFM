# =============================================================================
#
# =============================================================================


# allows to import own functions
import sys
import os
import re
root_project = re.findall(r'(^\S*TFM-master)', os.getcwd())[0]
sys.path.append(root_project)

from src.utils.help_func import results_searchcv, make_train_val_test,\
    plot_predictions
from src.features.add_features import features_graph, features_pop
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
from scipy.stats import randint
from yellowbrick.model_selection import LearningCurve, FeatureImportances
from yellowbrick.regressor import ResidualsPlot
from sklearn.metrics import r2_score
import joblib
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()

PATH =  f"{root_project}/models/decision_tree_rev17.pkl"


df = pd.read_csv(
    f'{root_project}/data/processed/simulation_results_rev17_wide.csv')
df = features_graph(df)
df = features_pop(df)



# df['total_deceased'] = np.log(df['total_deceased'].replace(0,np.nan))
# df['total_deceased'].fillna(0, inplace=True)


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



df = df[features]

X_train_val, y_train_val, X_test, y_test = make_train_val_test(df, out_mode=1)



param_dist = dict(
    max_depth=randint(low=8, high=18),
    min_samples_leaf=randint(2, 20),
)

random_search = RandomizedSearchCV(DecisionTreeRegressor(random_state=42),
                                   param_distributions=param_dist, verbose=2,
                                   n_iter=50, 
                                   random_state=42, n_jobs=-1)

random_search.fit(X_train_val, y_train_val)
joblib.dump(random_search, PATH)

# Load the model in path
random_search = joblib.load(PATH)

results_searchcv(random_search, X_test, y_test)

# y_predicted = random_search.predict(X_test)
# r_squared = r2_score(np.exp(y_test), np.exp(y_predicted))
# print(f"R2: {r_squared}"+"="*20)


fig, ax = plt.subplots(1, 1, figsize = (10,5))
visualizer = LearningCurve(random_search.best_estimator_, scoring='r2')
visualizer.fit(X_train_val, y_train_val)        # Fit the data to the visualizer
visualizer.show()           # Finalize and render the figure

fig, ax = plt.subplots(1, 1, figsize = (10,5))
viz = FeatureImportances(random_search.best_estimator_)
viz.fit(X_train_val, y_train_val)
viz.show()


fig, ax = plt.subplots(1, 1, figsize = (10,5))
viz = ResidualsPlot(random_search.best_estimator_)
viz.fit(X_train_val, y_train_val)
viz.score(X_test, y_test)
viz.show()


plot_predictions(random_search, X_test, y_test)


# errors_distribution(random_search, X_test, y_test, X_train_val)
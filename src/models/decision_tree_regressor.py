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
import joblib
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()
import time



# Read data
df = pd.read_csv(
    f'{root_project}/data/processed/simulation_results_rev17_wide.csv')
# Load features
df = features_graph(df)
df = features_pop(df)

# keep track of the original dataset
df_model = df.copy()

# size_data = 10000 # enter  desired subset of data
# df_model = df_model.sample(size_data)

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



df_model = df_model[features]

samples = df_model.shape[0]
features = df_model.shape[1]
run_time = time.strftime("run_%d_%m_%Y-%H_%M_%S")
MODEL_NAME = 'decision_tree_rev17'
# Path to save the model
PATH = f"{root_project}/models/{MODEL_NAME}-{samples}-samples-{features}-feat-{run_time}.pkl"

X_train_val, y_train_val, X_test, y_test = make_train_val_test(df_model, out_mode=1)



param_dist = dict(
    max_depth=randint(low=8, high=18),
    min_samples_leaf=randint(10, 30),
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
visualizer = LearningCurve(random_search.best_estimator_)
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
# =============================================================================
#
# =============================================================================

# allows to import own functions
import sys
import os
import re
root_project = re.findall(r'(^\S*TFM)', os.getcwd())[0]
sys.path.append(root_project)

from src.utils.help_func import results_searchcv,plot_predictions,\
    errors_distribution, plot_visualizations, get_model_data
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import clone
from scipy.stats import randint
import joblib
import seaborn as sns
sns.set()
import time
from scipy.stats import randint

# Get the data
df_train_val = get_model_data()

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

# Path naming
samples = df_train_val.shape[0]
features = df_train_val.shape[1]
run_time = time.strftime("run_%d_%m_%Y-%H_%M_%S")
MODEL_NAME = 'decision_tree'
# Path to save the model
PATH = f"{root_project}/models/tests/{MODEL_NAME}-{samples}-samples-{features}-feat-{run_time}"
LOAD_PATH = f"{root_project}/models/{MODEL_NAME}.pkl"
MODEL_PATH = f"{PATH}/{MODEL_NAME}.pkl"
RESULTS_PATH = f"{PATH}/results.txt"

if not os.path.exists(PATH):
    os.makedirs(PATH)


pipe = Pipeline([
    ('imputer', SimpleImputer()),
    ('estimator', DecisionTreeRegressor(random_state=42))
])

param_dist = dict(
    imputer__strategy=['mean','median'],
    estimator__max_depth=randint(low=8, high=25),
    estimator__min_samples_leaf=randint(8, 30),
)

scoring = {'R2': 'r2', 'RMSE': 'neg_root_mean_squared_error',
           'MAE': 'neg_mean_absolute_error'}

random_search = RandomizedSearchCV(pipe,
                                   param_distributions=param_dist,
                                   scoring=scoring,
                                   refit='R2',                                          
                                   verbose=1, n_iter=100, cv=3, n_jobs=-1)


random_search.fit(X_train_val, y_train_val)
joblib.dump(random_search, MODEL_PATH)

# Load a model
# random_search = joblib.load(LOAD_PATH)

results_searchcv(random_search, RESULTS_PATH)

pipe_plot = clone(pipe) # prevents yellowbrics from change pipe
plot_visualizations(PATH, pipe_plot, X_train,
                    y_train, X_train, y_train, X_val, y_val )

# Train the model with only train data and best parameters of random search
pipe.set_params(**random_search.best_params_)
pipe.fit(X_train, y_train)


plot_predictions(pipe, X_val, y_val, samples=50)

errors_distribution(pipe, X_val, y_val, df_train_val, n=1000)
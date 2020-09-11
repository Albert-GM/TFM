# =============================================================================
#
# =============================================================================


# allows to import own functions
import sys
import os
import re
root_project = re.findall(r'(^\S*TFM)', os.getcwd())[0]
sys.path.append(root_project)

from src.utils.help_func import plot_predictions,\
    errors_distribution, plot_visualizations, get_model_data,\
        results_searchcv_bayes
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import clone
import joblib
import seaborn as sns
sns.set()
import time
import numpy as np
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from skopt.callbacks import DeltaYStopper


# Get the data
df_train_val = get_model_data(200000)
seed = 42


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
                                                  random_state=seed)

# Path naming
samples = df_train_val.shape[0]
features = df_train_val.shape[1]
run_time = time.strftime("run_%d_%m_%Y-%H_%M_%S")
MODEL_NAME = 'random_forest'
# Path to save the model
PATH = f"{root_project}/models/tests/{MODEL_NAME}-{samples}-samples-{features}-feat-{run_time}"
LOAD_PATH = f"{root_project}/models/{MODEL_NAME}.pkl"
MODEL_PATH = f"{PATH}/{MODEL_NAME}.pkl"
RESULTS_PATH = f"{PATH}/results.txt"
if not os.path.exists(PATH):
    os.makedirs(PATH)
    


pipe = Pipeline([
    ('imputer', SimpleImputer()),
    ('estimator', RandomForestRegressor(random_state=seed))
])


max_depth = [int(x) for x in np.linspace(10, 60, num=11)]
max_depth.append(None)
search_space = dict(
    imputer__strategy=Categorical(['median', 'mean']),
    estimator__n_estimators=Integer(1,1000),
    estimator__max_features=Categorical(['auto', 'sqrt']),
    estimator__max_depth=max_depth,
    estimator__min_samples_split=Integer(2, 11),
    estimator__min_samples_leaf=Integer(1, 5),
    estimator__bootstrap=Categorical([True, False])
)

scores = []
i = 0
def on_step(optim_result, n_last=10):
    """
    Callback meant to view scores after each iteration while performing Bayesian
    Optimization in Skopt. Stops the optimization if the score in the last
    n_last iterations are equal."""
    global i
    i += 1
    scores.append(opt.best_score_)
    print(f"best score: {scores[-1]}")
    if i > n_last and len(set(scores[i-n_last:])) <= 1:
        return True
    
opt = BayesSearchCV(pipe, search_space, n_iter=32, cv=3, n_jobs=-1)

# Uncomment next lines to train the model
start= time.time()
opt.fit(X_train_val, y_train_val, callback=on_step)
joblib.dump(opt, MODEL_PATH)
print(f"Training time: {time.time() - start} seconds")
print(f"Best score cross-val: {opt.best_score_}")


# Load a model
# opt = joblib.load(LOAD_PATH)


results_searchcv_bayes(opt, RESULTS_PATH)


pipe.set_params(**opt.best_params_)



pipe_plot = clone(pipe) # prevents yellowbrics from change pipe
plot_visualizations(PATH, pipe_plot, X_train_val,
                    y_train_val, X_train, y_train, X_val, y_val )

# Train the pipe with only train data and best parameters of random search
pipe.fit(X_train, y_train)


plot_predictions(pipe, X_val, y_val, samples=50)

errors_distribution(pipe, X_val, y_val, df_train_val, n=1000)

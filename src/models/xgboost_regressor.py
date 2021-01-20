# allows to import own functions
import sys
import os
import re
root_project = re.findall(r'(^\S*TFM)', os.getcwd())[0]
sys.path.append(root_project)

from src.utils.help_func import plot_predictions,\
    errors_distribution, plot_visualizations, get_model_data, results_estimator,\
    results_searchcv_bayes
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.base import clone
import pandas as pd
import joblib
import seaborn as sns
sns.set()
import time
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from skopt.callbacks import DeltaYStopper

# Get the data
df_train_val = get_model_data()
seed=42

# Feature selection
features = [
    'R0',
    'omega',
    'Tc',
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
feat = df_train_val.shape[1]
run_time = time.strftime("run_%d_%m_%Y-%H_%M_%S")
MODEL_NAME = 'xgboost'
# Path to save the model
PATH = f"{root_project}/models/tests/{MODEL_NAME}-{samples}-samples-{feat}-feat-{run_time}"
LOAD_PATH = f"{root_project}/models/{MODEL_NAME}.pkl"
MODEL_PATH = f"{PATH}/{MODEL_NAME}.pkl"
RESULTS_PATH = f"{PATH}/results.txt"
if not os.path.exists(PATH):
    os.makedirs(PATH)
    


search_space = dict(
    n_estimators=Integer(15, 30),
    max_depth=Integer(5, 20),
    learning_rate=Real(0.01, 1),
    subsample=Real(0.6, 1),
    colsample_bytree=Real(0.6, 1),
    gamma=Integer(0,3)
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

    
opt = BayesSearchCV(xgb.XGBRegressor(random_state=seed), search_space,
                    n_iter=32, cv=3, n_jobs=-1)

# Uncomment next lines to train the model
start= time.time()
opt.fit(X_train_val, y_train_val, callback=on_step)
joblib.dump(opt, MODEL_PATH)
print("="*20)
print(f"Training time: {time.time() - start} seconds")
print(f"Best score cross-val: {opt.best_score_}")
print("="*20)


# Load a model
# opt = joblib.load(LOAD_PATH)


results_searchcv_bayes(opt, RESULTS_PATH)


estimator = xgb.XGBRegressor(**opt.best_params_, random_state=seed)



estimator_plot = clone(estimator) # prevents yellowbrics from change pipe
plot_visualizations(PATH, estimator_plot, X_train_val,
                    y_train_val, X_train, y_train, X_val, y_val )

# Train the pipe with only train data and best parameters of random search
estimator.fit(X_train, y_train)


plot_predictions(estimator, X_val, y_val, samples=50)

errors_distribution(estimator, X_val, y_val, df_train_val, n=1000)


# Score in test set
df_test = pd.read_pickle(
    f"{root_project}/data/processed/test_set.pickle")

df_test = df_test[features]

X_test = df_test.drop('total_deceased', axis=1)
y_test = df_test['total_deceased']


results_estimator(opt.best_estimator_, X_test, y_test)
plot_predictions(opt.best_estimator_, X_test, y_test, samples=50)



























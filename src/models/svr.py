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
from sklearn.model_selection import  RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from src.utils.help_func import results_searchcv,plot_predictions,\
    errors_distribution, plot_visualizations, get_model_data
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from yellowbrick.regressor import ResidualsPlot
from yellowbrick.model_selection import LearningCurve
import joblib
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()
import time


# Get the data
df_train_val = get_model_data(10000)

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
MODEL_NAME = 'support_vector_regessor'
# Path to save the model
PATH = f"{root_project}/models/{MODEL_NAME}-{samples}-samples-{features}-feat-{run_time}"
LOAD_PATH = f"{root_project}/models/{MODEL_NAME}.pkl"
if not os.path.exists(PATH):
    os.makedirs(PATH)
    

pipe = Pipeline([
    ('imputer', SimpleImputer()),
    ('preprocess', StandardScaler()),
    ('estimator', SVR())
])

param_dist = dict(
    imputer__strategy=['median', 'mean'],
    estimator__kernel = ['rbf'],
    estimator__C= loguniform(10, 2000),
    estimator__gamma= loguniform(1e-8, 1e-1)
)

scoring = {'R2': 'r2', 'RMSE': 'neg_root_mean_squared_error',
           'MAE': 'neg_mean_absolute_error'}

random_search = RandomizedSearchCV(pipe, param_distributions=param_dist,
                                   scoring=scoring,
                                   refit='R2',                                                   
                                   verbose=1, n_iter=50, cv=3,
                                   random_state=42, n_jobs=-1)


# random_search.fit(X_train_val, y_train_val)
# joblib.dump(random_search, f"{PATH}/{MODEL_NAME}.pkl")

# Load a model
random_search = joblib.load(LOAD_PATH)

# Train the pipe with only train data and best parameters of random search
pipe.set_params(**random_search.best_params_)
pipe.fit(X_train, y_train)

results_searchcv(random_search, pipe, X_val, y_val)

plot_visualizations(PATH, pipe, X_train_val,
                    y_train_val, X_val, y_val, featureimportance=False )

plot_predictions(pipe, X_val, y_val, samples=50)

errors_distribution(pipe, X_val, y_val, df_train_val, n=100)



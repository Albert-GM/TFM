# allows to import own functions
import sys
import os
import re
root_project = re.findall(r'(^\S*TFM-master)', os.getcwd())[0]
sys.path.append(root_project)

from src.utils.help_func import results_searchcv,plot_predictions,\
    errors_distribution, plot_visualizations, get_model_data
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, train_test_split
import pandas as pd
from scipy.stats import randint
import joblib
import seaborn as sns
sns.set()
import time
from scipy.stats import uniform, expon, randint, truncexpon, loguniform

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
                                                  y_train_val)

# Path naming
samples = df_train_val.shape[0]
features = df_train_val.shape[1]
run_time = time.strftime("run_%d_%m_%Y-%H_%M_%S")
MODEL_NAME = 'xgboost_rev17'
# Path to save the model
PATH = f"{root_project}/models/{MODEL_NAME}-{samples}-samples-{features}-feat-{run_time}"
if not os.path.exists(PATH):
    os.makedirs(PATH)
    

param_dist = dict(
    n_estimators=randint(low=15, high=30),
    max_depth=randint(low=5, high=20),
    learning_rate=loguniform(0.01, 1),
    subsample=uniform(loc=0.6, scale=1-0.8),
    colsample_bytree=uniform(loc=0.6, scale=1-0.8),
    gamma=[0, 1, 2]
)



random_search = RandomizedSearchCV( xgb.XGBRegressor(random_state=42),
                                   param_distributions=param_dist,
                                   verbose=1, n_iter=50, cv=3, n_jobs=-1)


random_search.fit(X_train_val, y_train_val)
joblib.dump(random_search, f"{PATH}/model.pkl")

# # Load a model
# # random_search = joblib.load(PATH)

# Train the model with only train data and best parameters of random search
estimator = xgb.XGBRegressor(**random_search.best_params_, random_state=42)
estimator.fit(X_train, y_train)

results_searchcv(random_search, estimator, X_val, y_val)

plot_visualizations(PATH, random_search.best_estimator_, X_train_val,
                    y_train_val, X_val, y_val )

plot_predictions(estimator, X_val, y_val, samples=50)

errors_distribution(estimator, X_val, y_val, df_train_val, n=1000 )


































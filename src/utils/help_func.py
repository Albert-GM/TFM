# =============================================================================
# Own functions that facilitate repetitive actions
# =============================================================================


import pandas as pd
import numpy as np
import networkx as nx
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import os
import re
from sklearn.pipeline import Pipeline
from yellowbrick.model_selection import LearningCurve, FeatureImportances
from yellowbrick.regressor import ResidualsPlot, PredictionError
root_project = re.findall(r'(^\S*TFM)', os.getcwd())[0]
# Uncomment following line when estimator is a keras sequential model
# from tensorflow.keras.models import Sequential


def extract_indicator(df, indicator, initial_year=None, final_year=None):
    """
    Selects the indicator and the years or year we are
    interested about. If intial year and final year is passed, the function
    returns the information from intial year to final year.If not initial year
    and final year is passed, the function returns the information of all the
    years. If final year is not passed, the function return the information
    only of one year (initial year).

    Parameters
    ---------
    df : pandas.DataFrame
        Dataframe containing the world indicators by country and year
    indicator : string
        Name of the indicador we want to exctract the information
    initial_year : int
    final_year : in

    Returns
    --------
    pandas.DataFrame
        Dataframe with the information about the indicador with the country
        name as row index and years as columns

    """

    if initial_year is None and final_year is None:
        new_df = df.loc[df['Indicator Name'] == indicator].drop(
            ['Indicator Name', 'Indicator Code'], axis=1)
        return new_df.reset_index(drop=True)
    elif final_year is None:
        keep_columns = ['Country Name', 'Country Code', str(initial_year)]
        new_df = df.loc[df['Indicator Name'] == indicator, keep_columns]
        new_df.rename(columns={str(initial_year): indicator}, inplace=True)
        return new_df.reset_index(drop=True)
    else:
        keep_columns = ['Country Name', 'Country Code'] + \
            list(range(initial_year, final_year + 1))
        keep_columns = [str(x) for x in keep_columns]
        new_df = df.loc[df['Indicator Name'] == indicator, keep_columns]
        return new_df.reset_index(drop=True)


def add_countryandloc(df, df_continents, df_coord):
    """
    Adds to a dataframe columns with the Continent Name, Continent Code;
    Longitude and Latitude of the country

    """
    df_continents.rename(
        columns={
            'Three_Letter_Country_Code': 'Country Code'},
        inplace=True)
    df_continents = df_continents.loc[:, [
        'Continent_Name', 'Continent_Code', 'Country Code']]
    df = pd.merge(df, df_continents, on='Country Code', how='inner')
    df = pd.merge(df, df_coord, on='Country Code', how='inner')
    return df


def last_values(df):
    """
    Provides a dataframe with the last values available by country. The
    dataframe has country as index and years as columns. First is necessary to
    apply extract_indicator to the original dataframe from world_indicators.

    """
    years = list(range(1960, 2020))
    years = [str(x) for x in years]

    last_values = []

    for country in df.iterrows():
        s = country[1]
        for year in reversed(years):
            # print(s[year])
            if not pd.isnull(s[year]):
                last_values.append(s[year])
                flag = True
                break
            elif year == '1960':
                last_values.append(np.nan)
            else:
                pass
    df2 = df.copy()
    df2['last_value'] = last_values
    return df2[['Country Name', 'Country Code', 'last_value']]


def results_searchcv(
        cv_estimator,
        path=None,
        estimator=None,
        X_test=None,
        y_test=None):
    """
    Prints out useful information about a cross-validated estimator. Given X_test,
    and y_test, provides performance in data not seen by the estimator. If path
    is not None, write the dictionary in a file.

    Parameters
    ----------
    cv_estimator : RandomizedSearchCV or GridSeachCV
        A trained cross-validated estimator.
    estimator : sklearn estimator
        A trained estimator, but not in X_test, y_test
    path : string
        Path to save the dictionary
    X_test : pandas.DataFrame or array
    y_test : pandas.DataFrame, pandas.Series or array

    Returns
    -------
    Dictionary with the information.

    """
    res_dict = {}
    res_dict['best_score_cross-val'] = cv_estimator.best_score_
    res_dict['std_cross-val'] = cv_estimator.cv_results_['std_test_R2'][cv_estimator.best_index_]
    res_dict['RMSE_cross-val'] = - \
        cv_estimator.cv_results_['mean_test_RMSE'][cv_estimator.best_index_]
    res_dict['MAE_cross-val'] = - \
        cv_estimator.cv_results_['mean_test_MAE'][cv_estimator.best_index_]
    res_dict['best_params'] = cv_estimator.best_params_
    if estimator is not None:
        y_predicted = estimator.predict(X_test)
        res_dict['R2_test'] = r2_score(y_test, y_predicted)
        res_dict['RMSE_test'] = mean_squared_error(
            y_test, y_predicted, squared=False)
        res_dict['MAE_test'] = mean_absolute_error(y_test, y_predicted)

    print("=" * 20)
    print(f"Cross-val best score:\n{res_dict['best_score_cross-val']}")
    print(
        f"Cross-val std:\n{res_dict['std_cross-val']}")
    print(f"Cross-val RMSE:\n{res_dict['RMSE_cross-val']}")
    print(f"Cross-val MAE:\n{res_dict['MAE_cross-val']}")
    print(f"Best parameters found:\n{res_dict['best_params']}")
    if estimator is not None:
        print(f"R-squared in test\n{res_dict['R2_test']}")
        print(
            f"RMSE in test:\n{res_dict['RMSE_test']}")
        print(f"MAE in test:\n{res_dict['MAE_test']}")
    print("=" * 20)

    if path is not None:
        with open(path, 'w') as file:
            file.write(json.dumps(res_dict))
    return res_dict


def results_searchcv_bayes(
        cv_estimator,
        path=None,
        estimator=None,
        X_test=None,
        y_test=None):
    """
    Prints out useful information about a cross-validated estimator. Given X_test,
    and y_test, provides performance in data not seen by the estimator. If path
    is not None, write the dictionary in a file.

    Parameters
    ----------
    cv_estimator : RandomizedSearchCV or GridSeachCV
        A trained cross-validated estimator.
    estimator : sklearn estimator
        A trained estimator, but not in X_test, y_test
    path : string
        Path to save the dictionary
    X_test : pandas.DataFrame or array
    y_test : pandas.DataFrame, pandas.Series or array

    Returns
    -------
    Dictionary with the information.

    """
    res_dict = {}
    res_dict['best_score_cross-val'] = cv_estimator.best_score_
    res_dict['std_cross-val'] = cv_estimator.cv_results_['std_test_score'][cv_estimator.best_index_]
    res_dict['best_params'] = cv_estimator.best_params_
    if estimator is not None:
        y_predicted = estimator.predict(X_test)
        res_dict['R2_test'] = r2_score(y_test, y_predicted)
        res_dict['RMSE_test'] = mean_squared_error(
            y_test, y_predicted, squared=False)
        res_dict['MAE_test'] = mean_absolute_error(y_test, y_predicted)

    print("=" * 20)
    print(f"Cross-val best score:\n{res_dict['best_score_cross-val']}")
    print(
        f"Cross-val std:\n{res_dict['std_cross-val']}")

    print(f"Best parameters found:\n{res_dict['best_params']}")
    if estimator is not None:
        print(f"R-squared in test\n{res_dict['R2_test']}")
        print(
            f"RMSE in test:\n{res_dict['RMSE_test']}")
        print(f"MAE in test:\n{res_dict['MAE_test']}")
    print("=" * 20)

    if path is not None:
        with open(path, 'w') as file:
            file.write(json.dumps(res_dict))
    return res_dict



def results_estimator(estimator, X_test, y_test):

    y_predicted = estimator.predict(X_test)
    print(f"R2 in test\n{r2_score(y_test, y_predicted)}")
    print(
        f"RMSE in test:\n{mean_squared_error(y_test, y_predicted, squared=False)}")
    print(f"MAE in test:\n{mean_absolute_error(y_test, y_predicted)}")
    return None


def top_k_connected(df, k):
    """
    Provides the k countries with higher degree

    Parameters
    ----------
    df : pandas.DataFrame
    k : int
        Number of countries to return

    Returns
    -------
    pandas.DataFrame
        k countries with higher degree.

    """

    graph = nx.read_gpickle(
        f"{root_project}/data/interim/routes_countries.gpickle")

    with open(f"{root_project}/data/interim/alpha3_to_alpha2.txt", 'r') as file:
        alpha3_to_alpha2 = json.load(file)

    degree = dict(nx.degree_centrality(graph))
    df['iso2'] = df['country_code'].map(alpha3_to_alpha2)
    df['degree'] = df['iso2'].map(degree)

    return df.sort_values(by='degree', ascending=False)[
        'country_code'].iloc[:k].tolist()


def make_train_val_test(df, test_val_prop=0.2, out_mode=0, shuffle=True):
    """
    Makes a train, validation and test sets according to the desired proportion
    for the test and validation sets. Validation set and test set are the same
    size. If prefered it can returns the train and validation sets together
    to use a cross validation method.

    Parameters
    ----------
    df : pandas.DataFrame
    test_val_prop : int, optional
    out_mode : int
        0 or 1. 0 if all three sets are wanted. 1 if train and validations sets
        are wanted together.

    Returns
    -------
    X_train : pandas.DataFrame
    X_val : pandas.DataFrame
    X_test : pandas.DataFrame
    y_train : pandas.Series
    y_val : pandas.Series
    y_test : pandas.Series

    """

    X = df.drop('total_deceased', axis=1)
    y = df['total_deceased']

    train_val_size = int(df.shape[0] * test_val_prop)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=train_val_size, random_state=42, shuffle=shuffle)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=train_val_size, random_state=42,
        shuffle=shuffle)

    print("=" * 20)
    if out_mode == 0:
        print(f"Train set: {X_train.shape}")
        print(f"Validation set: {X_val.shape}")
        print(f"Test set: {X_test.shape}")
        print("=" * 20)
        return X_train, X_val, X_test, y_train, y_val, y_test
    elif out_mode == 1:
        print(f"Train_validation set: {X_train_val.shape}")
        print(f"Test set: {X_test.shape}")
        print("=" * 20)
        return X_train_val, y_train_val, X_test, y_test
    else:
        raise ValueError('Incorrect out_mode value.')


def errors_distribution(estimator, X_val, y_val, df, n=200,
                        figsize=(20, 20)):
    """
    Prints out some plots to compare the distribution of the features in the
    train set against the distribuition of the features in the n samples with
    higher absoulute value prediction errors. It allows to see in which type
    of samples the model is predicting worse.

    Parameters
    ----------
    estimator : sklearn.estimator
        A trained sklearn.estimator.
    X_val : pandas.DataFrame
        Scaled if needed.
    y_val : pandas.DataFrame
    df : pandas.DataFrame
    n : int, optional
        Number of samples to consider in the errors set top. The default is 200.


    Returns
    -------
    None.

    """

    X_err = X_val.copy()

    # Uncomment following twho lines when estimator is a keras sequential model
    # if isinstance(estimator, Sequential):
    #     X_err = pd.DataFrame(X_err)

    X_err['predicted'] = estimator.predict(X_val).flatten()

    X_err['real'] = y_val
    X_err['error'] = X_err['real'] - X_err['predicted']
    X_err['abs_error'] = np.abs(X_err['error'])
    X_err_sorted = X_err.sort_values(by='abs_error', ascending=False).iloc[:n]
    error_idx = X_err_sorted.index

    number_subplots = len(df.describe().columns)
    width = 4
    high = int(number_subplots / 4)
    if number_subplots % 4 != 0:
        high = int(high + 1)

    fig, ax = plt.subplots(high, width, figsize=figsize)

    ax = ax.ravel()

    for i, feature in enumerate(df.describe().columns):
        sns.distplot(df[feature], hist=True, color='skyblue',
                     label='Original', ax=ax[i])
        sns.distplot(df.loc[error_idx, feature], hist=True, color='red',
                     label='Errors', ax=ax[i])

    return None


def plot_predictions(estimator, X_test, y_test, samples=20):
    """
    Plot predictions agains real values. It shows the number of predicted
    samples specified in the input.

    Parameters
    ----------
    estimator : sklearn.estimator
    X_test : pandas.DataFrame
    y_test : pandas.series
    samples : int, optional
        Samples to plot. The default is 20.

    Returns
    -------
    None.

    """

    y_predicted = estimator.predict(X_test).flatten()
    df_predicted = pd.DataFrame({'Actual': y_test, 'Predicted': y_predicted})
    df_predicted.sample(samples).plot(kind='barh', figsize=(10, 20))
    plt.show()

    return None


def plot_visualizations(PATH,
                        estimator,
                        X_train_val,
                        y_train_val,
                        X_train,
                        y_train,
                        X_val,
                        y_val,
                        figsize=(12, 8),
                        learningcurve=True,
                        predictionerror=True,
                        featureimportance=True,
                        residualsplot=True):
    """
    Plots differents plots that provide information about the estimator.

    Parameters
    ----------
    PATH : string
    estimator : sklearn.estimator
    X_train : pandas.DataFrame
    y_train : pandas.series
    X_val : pandas.DataFrame
    y_val : pandas.series
    figsize : tuple, optional
    learningcurve : TYPE, optional
        True if plot is wanted. The default is True.
    featureimportance : TYPE, optional
        True if plot is wanted. The default is True.
    residualsplot : TYPE, optional
        True if plot is wanted. The default is True.

    Returns
    -------
    None.

    """

    if learningcurve:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        visualizer = LearningCurve(estimator, n_jobs=-1)
        # Fit the data to the visualizer
        visualizer.fit(X_train_val, y_train_val)
        visualizer.show()           # Finalize and render the figure
        plt.savefig(
            f"{PATH}/learning_curve.png")
        
    if predictionerror:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        viz = PredictionError(estimator)
        viz.fit(X_train, y_train)
        viz.score(X_val, y_val)
        viz.show()
        plt.savefig(
            f"{PATH}/prediction_error.png")        

    if residualsplot:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        viz = ResidualsPlot(estimator)
        viz.fit(X_train, y_train)
        viz.score(X_val, y_val)
        viz.show()
        plt.savefig(
            f"{PATH}/residuals.png")

    if featureimportance:
        if isinstance(estimator, Pipeline):
            estimator = estimator['estimator']
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        viz = FeatureImportances(estimator)
        viz.fit(X_train_val, y_train_val)
        viz.show()
        plt.savefig(
            f"{PATH}/feature_importance.png")

    return None


def take_samples(df_v1, df_v2, n_samples, ratio_errors=0.2):
    """
    Mixs two dataframes according to the ratio enter as input.

    Parameters
    ----------
    df_v1 : pandas.DataFrame
    df_v2 : pandas.DataFrame
    n_samples : int
        Total number of samples in the final dataframe.
    ratio_errors : float, optional
        Ratio of df_v2 to include in the final dataframe. The default is 0.2.


    Returns
    -------
    df : pandas.DataFrame

    """

    if df_v1.shape[1] != df_v2.shape[1]:
        raise ValueError("Data have different number of features.")

    l1 = df_v1.shape[0]
    l2 = df_v2.shape[0]
    total_l = l1 + l2

    error_samples = int(n_samples * ratio_errors)
    normal_samples = int(n_samples - error_samples)

    if error_samples > l2:
        raise ValueError("Not enough sample from errors distributions.")
    if normal_samples > l1:
        raise ValueError("Not enough sample from original distribution.")
    else:
        df = pd.concat([df_v1.sample(normal_samples, random_state=42),
                        df_v2.sample(error_samples, random_state=42)],
                       ignore_index=True).sample(
                           frac=1, random_state=42).reset_index(drop=True)
    return df


def get_model_data(n_samples=None, ratio=None):
    """
    Provides train and validation data to train the model. If n_samples and
    ratio are not None, it returns data according to the ratio between v1 and v2.
    V1 is data comming from the original distribution of SIRD parameters, and
    V2 is data comming from distributions based on errors of trained ML models.

    Parameters
    ----------
    n_samples : int, optional
        Subset of samples from the original set. The default is None.
    ratio : float, optional
        Ratio of the data from distribudion based on errors. The default is None.

    Returns
    -------
    df_train_val : pandas.DataFrame

    """

    df_train_val = pd.read_pickle(
        f"{root_project}/data/processed/train_val_set.pickle")
    df_train_val_rev = pd.read_pickle(
        f"{root_project}/data/processed/train_val_set_rev.pickle")
    df_v1_train_val = pd.read_pickle(
        f"{root_project}/data/processed/train_val_set_v1.pickle")
    df_v2_train_val = pd.read_pickle(
        f"{root_project}/data/processed/train_val_set_v2.pickle")

    if n_samples is not None and ratio is not None:
        df_train_val = take_samples(df_v1_train_val,
                                    df_v2_train_val,
                                    n_samples,
                                    ratio)
        return df_train_val
    elif n_samples is not None:
        df_train_val = df_train_val.sample(n_samples, random_state=42)
        return df_train_val
    else:
        return df_train_val_rev
    
    


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
root_project = re.findall(r'(^\S*TFM-master)', os.getcwd())[0]


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
    # flag = False
    # mejorar este codigo, puedo añadir una flag, para decirme si todos han
    # sido nan
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
    df2 = df.copy()  # para no cambiar el df del input
    df2['last_value'] = last_values
    return df2[['Country Name', 'Country Code', 'last_value']]


def results_searchcv(estimator, X_test=None, y_test=None):
    """
    Prints out useful information about a trained GridsearchCV object or a
    RandomizedSearchCV object from the sklearn library. Given a pair X_test,
    y_test prints out the score and the mean absolute error from the estimator.

    Parameters
    ----------
    predictor : sklearn.model_selection._search.GridSearchCV
        A trained predictor.
    X_test : pandas.DataFrame or array
    y_test : pandas.DataFrame, pandas.Series or array

    Returns
    -------
    None.

    """
    print("="*20)
    print(f"Cross-val best score:\n{estimator.best_score_}")
    print(f"Cross-val std:\n{estimator.cv_results_['std_test_score'][estimator.best_index_]}")
    print(f"Best parameters found:\n{estimator.best_params_}")
    if X_test is not None and y_test is not None:
        print(f"Score in test:\n{estimator.score(X_test, y_test)}")
        y_predicted = estimator.predict(X_test)
        print(f"R^2 in test\n{r2_score(y_test, y_predicted)}")
        print(f"MAE in test:\n{mean_absolute_error(y_test, y_predicted)}")
    print("="*20)





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


def make_train_val_test(df, test_val_prop=0.2, out_mode=0):
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

    df = df.iloc[::-1] # new simulation data at the beggining of df
    X = df.drop('total_deceased', axis=1)
    y = df['total_deceased']

    train_val_size = int(df.shape[0] * test_val_prop)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=train_val_size, random_state=42, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=train_val_size, random_state=42,
        shuffle=False)
    
    print("="*20)
    if out_mode==0:
        print(f"Train set: {X_train.shape}")
        print(f"Validation set: {X_val.shape}")
        print(f"Test set: {X_test.shape}")
        print("="*20)
        return X_train, X_val, X_test, y_train, y_val, y_test
    elif out_mode==1:
        print(f"Train_validation set: {X_train_val.shape}")
        print(f"Test set: {X_test.shape}")
        print("="*20)
        return X_train_val, y_train_val, X_test, y_test
    else:
        raise ValueError('Incorrect out_mode value.')


def errors_distribution(estimator, X_test, y_test, X_train, n=200,
                        X_test_scaled=None):
    """
    Prints out some plots to compare the distribution of the features in the
    train set against the distribuition of the features in the n samples with
    higher absoulute value prediction errors. It allows to see in which type
    of samples the model is predicting worse. If scaled_test='True' pass as
    extra argument X_test_scaled

    Parameters
    ----------
    model : sklearn.estimator
        A trained sklearn.estimator.
    X_test : pandas.DataFrame
    y_test : pandas.DataFrame
    X_train : pandas.DataFrame
        Without scaling.
    n : int, optional
        Number of samples to consider in the errors set top. The default is 200.
    X_test_scaled : bool, optional
        If passed the estimator predicts with scaled data.
        

    Returns
    -------
    None.

    """
    
    X_err = X_test.copy()
    if X_test_scaled is None:
        X_err['predicted'] = estimator.predict(X_test)
    else:
        X_err['predicted'] = estimator.predict(X_test_scaled)
          
    X_err['real'] = y_test
    X_err['error'] = X_err['real'] - X_err['predicted']
    X_err['abs_error'] = np.abs(X_err['error'])
    X_err_sorted = X_err.sort_values(by='abs_error', ascending=False).iloc[:n]

    for column in X_train.columns:
      fig, ax = plt.subplots(1, 1, figsize = (15,8))
      sns.distplot(X_train[column], hist=True, color='skyblue',
                   label='Original', ax=ax)
      sns.distplot(X_err_sorted[column], hist=True, color='red',
                   label='Errors', ax=ax)
      ax.set(title=column)
      plt.legend()
      plt.show()
      
    return None



def plot_predictions(estimator, X_test, y_test, samples=50):
    
    
    y_predicted = estimator.predict(X_test)
    df_predicted = pd.DataFrame({'Actual': y_test, 'Predicted': y_predicted})
    df_predicted.sample(samples).plot(kind='barh',figsize=(15,50))
    plt.show()
    
    return None












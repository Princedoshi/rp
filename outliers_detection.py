import streamlit as st
import pandas as pd
from scipy.stats import zscore, chi2
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

def detect_outliers_zscore(df, column, threshold=3):
    z_scores = (df[column] - df[column].mean()) / df[column].std()
    outliers = abs(z_scores) > threshold
    return outliers

def correct_outliers_zscore(df, column, threshold=3):
    outliers = detect_outliers_zscore(df, column, threshold)
    df.loc[outliers, column] = df[column].median() 
    return df

def detect_outliers_mahalanobis(df, columns):
    cov_matrix = df[columns].cov()
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    mahalanobis_distances = np.matmul(np.matmul((df[columns] - df[columns].mean()).values, inv_cov_matrix),
                                      (df[columns] - df[columns].mean()).values.T)
    threshold = chi2.ppf(0.975, df=len(columns))
    outliers = mahalanobis_distances > threshold
    return outliers

def correct_outliers_mahalanobis(df, columns):
    outliers = detect_outliers_mahalanobis(df, columns)
    df.loc[outliers, columns] = df[columns].median()
    return df

def detect_outliers_lof(df, columns, contamination=0.05):
    lof_model = LocalOutlierFactor(contamination=contamination)
    outliers = lof_model.fit_predict(df[columns]) == -1
    return outliers

def correct_outliers_lof(df, columns, contamination=0.05):
    outliers = detect_outliers_lof(df, columns, contamination)
    df.loc[outliers, columns] = df[columns].median()
    return df

def detect_outliers_iqr(df, column):
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr_value = q3 - q1
    lower_bound = q1 - 1.5 * iqr_value
    upper_bound = q3 + 1.5 * iqr_value
    outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
    return outliers

def correct_outliers_iqr(df, column):
    outliers = detect_outliers_iqr(df, column)
    df.loc[outliers, column] = df[column].median()
    return df

def detect_outliers_isolation_forest(df, columns, contamination=0.05):
    isolation_forest = IsolationForest(contamination=contamination)
    outliers = isolation_forest.fit_predict(df[columns]) == -1
    return outliers

def correct_outliers_isolation_forest(df, columns, contamination=0.05):
    outliers = detect_outliers_isolation_forest(df, columns, contamination)
    df.loc[outliers, columns] = df[columns].median()
    return df

def detect_outliers_dbscan(df, columns, eps=0.5, min_samples=5):
    dbscan_model = DBSCAN(eps=eps, min_samples=min_samples)
    outliers = dbscan_model.fit_predict(df[columns]) == -1
    return outliers

def correct_outliers_dbscan(df, columns, eps=0.5, min_samples=5):
    outliers = detect_outliers_dbscan(df, columns, eps, min_samples)
    df.loc[outliers, columns] = df[columns].median()
    return df

def detect_and_correct_outliers(df, column, method='auto'):
    if df[column].dtype in ['int64', 'float64']:
        if method == 'zscore' or (method == 'auto' and df[column].std() > 10):
            df = correct_outliers_zscore(df, column)

        elif method == 'mahalanobis' or (method == 'auto'):
            # Only apply Mahalanobis correction if there are numeric columns for correlation
            numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
            correlated_columns = numeric_columns[df[numeric_columns].corr()[column] > 0.8].tolist()
            if len(correlated_columns) > 1:
                df = correct_outliers_mahalanobis(df, correlated_columns)

        elif method == 'lof' or (method == 'auto'):
            df = correct_outliers_lof(df, [column])

        elif method == 'iqr' or (method == 'auto' and df[column].skew() > 1):
            df = correct_outliers_iqr(df, column)

        elif method == 'isolation_forest' or (method == 'auto'):
            df = correct_outliers_isolation_forest(df, [column])

        elif method == 'dbscan' or (method == 'auto'):
            df = correct_outliers_dbscan(df, [column])

    return df
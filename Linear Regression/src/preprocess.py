import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from .config import numerical_cols, categorical_cols

def feature_engineering(df):
    df = df.copy()
    df['AvgLength'] = df[['Length1', 'Length2', 'Length3']].mean(axis=1)
    df.drop(columns=['Length1', 'Length2', 'Length3'], inplace=True)
    return df

def get_preprocessor():
    return ColumnTransformer(transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
    ])

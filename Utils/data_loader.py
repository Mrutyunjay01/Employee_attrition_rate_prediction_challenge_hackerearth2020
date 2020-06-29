# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_df(filepath):
    """ Load dataset without Employee ID column."""
    df = pd.read_csv(filepath)
    df = df.drop(columns=['Employee_ID'], axis=1)

    return df
    pass


def replace_categorical(df):
    """ Apply label encoding instead."""
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = LabelEncoder().fit_transform(df[col])

    return df
    pass

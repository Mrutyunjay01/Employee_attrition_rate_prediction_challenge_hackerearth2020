# -*- coding: utf-8 -*-

import pandas as pd


def load_df(filepath):
    """ Load dataset without Employee ID column."""
    df = pd.read_csv(filepath)
    df = df.drop(columns=['Employee_ID'], axis=1)

    return df
    pass



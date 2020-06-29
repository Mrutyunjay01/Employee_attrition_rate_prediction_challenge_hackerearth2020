from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import argparse
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge

# parse arguments for loading and saving csv files
ap = argparse.ArgumentParser()
ap.add_argument('-r', '--train', required=True,
                help='Path to train.csv')
ap.add_argument('-t', '--test', required=True,
                help='Path to test.csv')
ap.add_argument('-s', '--submission', required=True,
                help='Path to submission.csv')
agrs = vars(ap.parse_args())


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


# load our train and test data
train_df = load_df(agrs['train'])
test_df = load_df(agrs['test'])
submission = pd.DataFrame()

# preprocess data

train_df = replace_categorical(train_df)
test_df = replace_categorical(test_df)


class RidgeTransformer(Ridge, TransformerMixin):

    def transform(self, X, *_):
        return self.predict(X).reshape(len(X), -1)


class RandomForestTransformer(RandomForestRegressor, TransformerMixin):

    def transform(self, X, *_):
        return self.predict(X).reshape(len(X), -1)


class KNeighborsTransformer(KNeighborsRegressor, TransformerMixin):

    def transform(self, X, *_):
        return self.predict(X).reshape(len(X), -1)


def build_model():
    ridge_transformer = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('poly_feats', PolynomialFeatures()),
        ('ridge', RidgeTransformer())
    ])

    pred_union = FeatureUnion(
        transformer_list=[
            ('ridge', ridge_transformer),
            ('rand_forest', RandomForestTransformer()),
            ('knn', KNeighborsTransformer())
        ],
        n_jobs=2
    )

    model = Pipeline(steps=[
        ('pred_union', pred_union),
        ('lin_regr', LinearRegression())
    ])

    return model


print('Build and fit a model...')

model = build_model()

model.fit(X=train_df.drop(['Attrition_rate'], axis=1), y=train_df['Attrition_rate'])
# predicting on the training set
predictions = model.predict(train_df.drop(['Attrition_rate'], axis=1))

print("RMSE : ", max(0, 1 - np.sqrt(mean_squared_error(train_df['Attrition_rate'], predictions))))

# parsing the testing data
print('Predicting on test file')
test_predicted = model.predict(test_df)

# creating submission file
submission['Employee_ID'] = pd.read_csv(agrs['test'])['Employee_ID']
submission['Attrition_rate'] = test_predicted
assert len(submission) == len(test_df)

print("Saving the submission file...")
submission.to_csv(agrs['submission'], index=False)


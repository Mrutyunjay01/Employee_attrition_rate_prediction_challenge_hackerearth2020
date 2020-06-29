from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import argparse

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

# instantiating XGB Regressor
xgbreg = xgb.sklearn.XGBRegressor()

# initialize parameters
parameters = {'colsample_bytree': 0.7, 'learning_rate': 0.01, 'max_depth': 4, 'min_child_weight': 6, 'n_estimators': 500, 'nthread': 4, 'silent': 1, 'subsample': 0.7}

# xgb_grid = GridSearchCV(xgbreg, parameters, n_jobs=5, cv=4, verbose=True)
xgb_grid = xgbreg.set_params(parameters=parameters)
# fit our training data
xgb_grid.fit(X=train_df.drop(['Attrition_rate'], axis=1), y=train_df['Attrition_rate'])

# optimised paramters
# print(xgb_grid.best_params_)

preditctions = xgb_grid.predict(train_df.drop(['Attrition_rate'], axis=1))
print('RMSE score : ', 1 - np.sqrt(mean_squared_error(train_df['Attrition_rate'], y_pred=preditctions)))

# parse test file
print('Predicting on test file')
test_predicted = xgb_grid.predict(test_df)

# creating submission file
submission['Employee_ID'] = pd.read_csv(agrs['test'])['Employee_ID']
submission['Attrition_rate'] = test_predicted
assert len(submission) == len(test_df)

print("Saving the submission file...")
submission.to_csv(agrs['submission'], index=False)


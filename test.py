import numpy as np
import pandas as pd
import pickle
import sys
import xgboost as xgb
from joblib import load
from preprocessing import data_cleaning, time_distance

def rmspe(preds, actuals):
    preds = preds.reshape(-1)
    actuals = actuals.reshape(-1)
    assert preds.shape == actuals.shape
    error = 100 * np.linalg.norm((actuals - preds) / actuals) / np.sqrt(preds.shape[0])
    return error

def feature_engineering(df):
    df["Year"] = df.Date.dt.year
    df["Month"] = df.Date.dt.month
    df["Day"] = df.Date.dt.day
    df["WeekOfYear"] = df.Date.dt.weekofyear
    df = time_distance(df, 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Comp_since_days')
    df = time_distance(df, 'Promo2SinceWeek', 'Promo2SinceYear','Promo_since_days', 1)
    test_features_eng = pickle.load(open("training/augmented_features.dat", "rb"))
    #§print(test_features_eng.keys())
    df["MedianSalesByStore"] = df.Store.map(test_features_eng["MedianSalesByStore"])
    df["MedianSalesByDayOfWeek"] = df.DayOfWeek.map(test_features_eng["MedianSalesByDayOfWeek"])
    df["Monetary"] = df.Store.map(test_features_eng["Monetary"])
    df["Recency"] = df.Store.map(test_features_eng["Recency"])
    df["Customer_avg"] = df.Store.map(test_features_eng["Customer_avg"])
    return df


def load_data():
    print("Loading test data.")
    df = pd.read_csv("data/test.csv", header=0, parse_dates=["Date"], dtype={"StateHoliday":object})
    print(str(df.shape[0]) + " rows have been found.")
    print("Loading store data.")
    store = pd.read_csv("data/store.csv", header=0)
    print(str(store.shape[0]) + " rows have been found.")
    print("Filter rows where Sales is bigger than zero")
    df = df[df.Sales > 0]
    print("Join train and store")
    df = pd.merge(df, store, on="Store")
    return df

test = load_data()
test = data_cleaning(test)
test = feature_engineering(test)

print("Loaded the model")
bst = load("training/rossman.dat")
Y_test = test.Sales.values
features = bst.feature_names
X_test = test[features]

print("Run predictions")
yhat_test = bst.predict(xgb.DMatrix(X_test))

rmspe_test = rmspe(yhat_test, Y_test)
print("RMSPE on test set: {:.3f}%".format(rmspe_test))

import numpy as np
import pandas as pd
import pickle
import sys
import xgboost as xgb
from joblib import load
from preprocessing import data_cleaning 

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
    test_features_eng = pickle.load(open("augmented_features.dat", "rb"))
    #§print(test_features_eng.keys())
    df["MedianSalesByStore"] = df.Store.map(test_features_eng["MedianSalesByStore"])
    df["MedianSalesByDayOfWeek"] = df.DayOfWeek.map(test_features_eng["MedianSalesByDayOfWeek"])
    df["Monetary"] = df.Store.map(test_features_eng["Monetary"])
    df["Recency"] = df.Store.map(test_features_eng["Recency"])
    df["Customer_avg"] = df.Store.map(test_features_eng["Customer_avg"])
    return df

def time_distance(df,sinceWorM,sinceYr,distance_name,week=None):
    # sinceWorM : string of column name with ...sinceMonth, if sinceWeek => week=1
    # sinceYr : string of column name with ... SinceYear
    # distance_name : ex. competition_dist or promo_dist
    m_999 = df[sinceWorM] != -999
    y_999 = df[sinceYr] != -999
    glob_999 = m_999 + y_999
    df_mask = df.loc[glob_999,:]
    if week == 1:
        df_mask.loc[:,sinceWorM]=df_mask.loc[:,sinceWorM] // 4
        df_mask.loc[:,sinceWorM] = df_mask.loc[:,sinceWorM].apply(str).replace({"0":"1","13":"12"})
    df_mask.loc[:,"temp_MY"] = pd.to_datetime(df_mask.loc[:,sinceYr].apply(str) + "-" + df_mask.loc[:,sinceWorM].apply(str) + "-01")
    df_mask.loc[:,distance_name] = (df_mask.loc[:,"Date"] - df_mask.loc[:,"temp_MY"]).dt.days
    # concat back columns with distance in days to original df
    new_df = pd.concat([df,df_mask.loc[:,distance_name]], axis=1, sort=False)
    new_df.loc[:,distance_name] = new_df.loc[:,distance_name].fillna(value=-9999).astype("int", copy=False)
    return new_df


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
bst = load("rossman.dat")
Y_test = test.Sales.values
features = bst.feature_names
X_test = test[features]

print("Run predictions")
yhat_test = bst.predict(xgb.DMatrix(X_test))

rmspe_test = rmspe(yhat_test, Y_test)
print("RMSPE on test set: {:.3f}%".format(rmspe_test))

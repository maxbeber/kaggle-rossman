import numpy as np
import pickle
import pandas as pd
import sys
import xgboost as xgb
from joblib import load
from sklearn.impute import SimpleImputer


def rmspe(preds, actuals):
    preds = preds.reshape(-1)
    actuals = actuals.reshape(-1)
    assert preds.shape == actuals.shape
    error = 100 * np.linalg.norm((actuals - preds) / actuals) / np.sqrt(preds.shape[0])
    return error


def data_cleaning(df):
    missing_value_numeric_features = ["Promo", "CompetitionDistance", "CompetitionOpenSinceMonth", \
      "CompetitionOpenSinceYear", "Promo2SinceYear", "Promo2SinceWeek", "SchoolHoliday"]
    integer_features = ["Open", "DayOfWeek", "Promo", "Store", "Customers", "CompetitionOpenSinceYear", \
      "CompetitionOpenSinceMonth", "Promo2SinceYear", "Promo2SinceWeek", "CompetitionDistance"]
    df.loc[:, missing_value_numeric_features] = df.loc[:, missing_value_numeric_features].fillna(value=-999)
    df.loc[:, "Open"].fillna(value="1", inplace=True)
    mean_imputer = SimpleImputer(strategy="median")
    df[["Customers"]] = mean_imputer.fit_transform(df[["Customers"]])
    df.loc[:, "DayOfWeek"] = df.Date.dt.dayofweek + 1
    categories = ["StateHoliday", "StoreType", "Assortment"]
    df.loc[:, categories].fillna(value="Missing", inplace=True)
    df.loc[:, "StateHoliday"] = df.StateHoliday.replace({"0.0": "0"})
    df[integer_features] = df[integer_features].astype("int64", copy=False)
    one_hot_encoding_features = ["SchoolHoliday", "StateHoliday", "StoreType", "Assortment", "PromoInterval"]
    df = pd.get_dummies(df, columns=one_hot_encoding_features)
    return df


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


print("Loading test data.")
test = pd.read_csv("data/test.csv", header=0, parse_dates=["Date"], dtype={"StateHoliday":object})
print(str(test.shape[0]) + " rows have been found.")

print("Loading store data.")
store = pd.read_csv("data/store.csv", header=0)
print(str(store.shape[0]) + " rows have been found.")

print("Filter rows where Sales is bigger than zero")
test = test[test.Sales > 0]

print("Join train and store")
test = pd.merge(test, store, on="Store")
test = data_cleaning(test)
test = feature_engineering(test)

Y_test = test.Sales
X_test = test.drop(columns=["Sales"], axis=1)

print("Loaded the model")
bst = load("rossman.dat")
#bst = xgb.Booster({"nthread": 4})
#bst.load_model("rossman.bin")

print("Run predictions")
yhat_test = bst.predict(xgb.DMatrix(X_test.values))

test_pred = yhat_test.values
test_actuals = Y_test.values
rmspe_test = rmspe(test_pred, test_actuals)
print("RMSPE on TEST SET: {:.3f}".format(rmspe_test))
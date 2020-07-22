import datetime
import numpy as np
import sys
import pandas as pd
import pickle
import xgboost as xgb
from joblib import dump
from metrics import rmspe
from preprocessing import data_cleaning, time_distance, m_group
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

features = [ \
    "Store", "DayOfWeek", "Customers", "Open", "Promo", \
    "CompetitionDistance","Comp_since_days", \
    "Promo2", 'Promo_since_days', \
    "StateHoliday_0", "StateHoliday_a", \
    "StateHoliday_b", "StateHoliday_c", "StoreType_a", "StoreType_b", \
    "StoreType_c", "StoreType_d", "Assortment_a", "Assortment_b", \
    "Assortment_c", "PromoInterval_Feb,May,Aug,Nov", \
    "PromoInterval_Jan,Apr,Jul,Oct", "PromoInterval_Mar,Jun,Sept,Dec", \
    "Year", "Month", "WeekOfYear","MedianSalesByStore", "MedianSalesByDayOfWeek", \
    "Monetary" , "Recency", "Customer_avg"]
params = {
    "objective": "reg:squarederror",
    "max_depth": 5,
    "nthread": 6
}
num_boost_round = 130
test_features_eng = {}


def feature_engineering(df):
    df["Year"] = df.Date.dt.year
    df["Month"] = df.Date.dt.month
    df["Day"] = df.Date.dt.day
    df["WeekOfYear"] = df.Date.dt.weekofyear
    df = time_distance(df, 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Comp_since_days')
    df = time_distance(df, 'Promo2SinceWeek', 'Promo2SinceYear','Promo_since_days', 1)
    median_sales_by_store = df.groupby("Store")["Sales"].median()
    # Convert to dictionnaire to use in TEST SET and Save it in global Dictionnary
    test_features_eng["MedianSalesByStore"] = median_sales_by_store.to_dict()
    df["MedianSalesByStore"] = df.Store.map(median_sales_by_store)
    median_sales_by_day_of_week = df.groupby("DayOfWeek")["Sales"].median()
    # Convert to dictionnaire to use in TEST SET and Save it in global Dictionnary
    test_features_eng["MedianSalesByDayOfWeek"] = median_sales_by_day_of_week.to_dict()
    df["MedianSalesByDayOfWeek"] = df.DayOfWeek.map(median_sales_by_store)
    ## RFM
    today = datetime.datetime(2014,8,1)
    agg_rule = {'Date': lambda x: (today - x.max()).days, 'Customers': lambda x: x.median(), 'Sales': lambda x: x.sum()}
    rfm_score = df.groupby('Store').agg(agg_rule)
    rfm_score.columns = ["R", "C", "M"]
    quantiles = rfm_score.quantile(q=[0.25, 0.5, 0.75])
    ## Recency Feature
    test_features_eng["Recency"] = rfm_score["R"].to_dict()
    df["Recency"] = df.Store.map(test_features_eng["Recency"])
    # Convert to dictionnaire to use in TEST SET and Save it in global Dictionnary
    ## Monetary Feature
    rfm_score["M_segments"] = rfm_score['M'].apply(m_group, args=('M',quantiles))
    # Convert to dictionnaire to use in TEST SET and Save it in global Dictionnary
    test_features_eng["Monetary"] = rfm_score["M_segments"].to_dict()
    df["Monetary"] = df.Store.map(test_features_eng["Monetary"])
    test_features_eng["Customer_avg"] = rfm_score["C"].to_dict()
    df["Customer_avg"] = df.Store.map(test_features_eng["Customer_avg"])
    pickle.dump(test_features_eng, open("training/augmented_features.dat", "wb"))
    return df


def load_data():
    train = pd.read_csv("data/train.csv", header=0, parse_dates=["Date"], dtype={"StateHoliday":object})
    print(str(train.shape[0]) + " rows have been found.")
    print("Loading store data.")
    store = pd.read_csv("data/store.csv", header=0)
    print(str(store.shape[0]) + " rows have been found.")
    print("Filter rows where Sales is bigger than zero")
    train = train[train.Sales > 0]
    print("Filter rows where Open is true")
    train = train[train.Open != 0]
    print("Join train and store")
    train = pd.merge(train, store, on="Store")
    return train


def display_features_by_importance(features):
    features_dict = {}
    feature_importance_sorted = sorted(features.items(), key=lambda kv: kv[1], reverse=True)
    for i in range(len(feature_importance_sorted)):
        feature, importance_rate = feature_importance_sorted[i]
        features_dict[feature] = importance_rate
    print(features_dict)


def display_metrics(x_train, y_train, x_holdout, y_holdout):
    print("")
    print("#################################################")
    train_pred = x_train["Prediction"].values
    train_actuals = y_train.values
    rmspe_train = rmspe(train_pred, train_actuals)
    print("RMSPE on TRAIN SET: {:.3f}%".format(rmspe_train))
    predictions = x_holdout["Prediction"].values
    actuals = y_holdout.values
    rmspe_test = rmspe(predictions, actuals)
    print("RMSPE on TEST SET: {:.3f}%".format(rmspe_test))
    print("#################################################")
    print("")


print("Loading training data.")
data = load_data()

print("Clean up data")
data = data_cleaning(data)

print("Feature engineering")
data = feature_engineering(data)

Y = data.Sales
X = data.drop(columns=["Date", "Sales"], axis=1)
X_train, X_holdout, Y_train, Y_holdout = train_test_split(X, Y, test_size=0.30, random_state=42, shuffle=True)

print("Train Random Forrest")
dtrain = xgb.DMatrix(X_train[features].values, label=Y_train.values, feature_names=features)
dvalid = xgb.DMatrix(X_holdout[features].values, label=Y_holdout.values, feature_names=features)
watchlist = [(dtrain, "train"), (dvalid, "eval")]
gbm = xgb.XGBRegressor()
gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=25, verbose_eval=True)

print("Computing predictions")
yhat_holdout = gbm.predict(xgb.DMatrix(X_holdout[features]))
X_holdout["Prediction"] = yhat_holdout
yhat_train = gbm.predict(xgb.DMatrix(X_train[features]))
X_train["Prediction"] = yhat_train

print("Display the list of features by importance")
feature_score = gbm.get_score(importance_type="gain")
display_features_by_importance(feature_score)

print("Save the model to rossman.dat")
dump(gbm, "training/rossman.dat")

display_metrics(X_train, Y_train, X_holdout, Y_holdout)
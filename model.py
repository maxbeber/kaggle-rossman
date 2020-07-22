import numpy as np
import sys
import pandas as pd
import xgboost as xgb
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


def metric(preds, actuals):
	preds = preds.reshape(-1)
	actuals = actuals.reshape(-1)
	assert preds.shape == actuals.shape
	return 100 * np.linalg.norm((actuals - preds) / actuals) / np.sqrt(preds.shape[0])


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
    average_sales_by_store = df.groupby("Store")["Sales"].median()
    df["AverageSalesByStore"] = df.Store.map(average_sales_by_store)
    average_sales_by_day_of_week = df.groupby("DayOfWeek")["Sales"].median()
    df["AverageSalesByDayOfWeek"] = df.DayOfWeek.map(average_sales_by_store)
    return df


print("Loading training data.")
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

X_train, X_holdout = train_test_split(train, test_size=0.30, random_state=42, shuffle=True)

print("Clean up data")
X_train = data_cleaning(X_train)
X_holdout = data_cleaning(X_holdout)

print("Feature engineering")
X_train = feature_engineering(X_train)
X_holdout = feature_engineering(X_holdout)
print(X_train.info())
print(X_holdout.info())

print("Train Random Forrest")
features = ["Store", "DayOfWeek", "Customers", "Open", "Promo", \
	"CompetitionDistance", "CompetitionOpenSinceMonth", \
	"CompetitionOpenSinceYear", "Promo2", "Promo2SinceWeek", \
	"Promo2SinceYear", "StateHoliday_0", "StateHoliday_a", \
	"StateHoliday_b", "StateHoliday_c", "StoreType_a", "StoreType_b", \
	"StoreType_c", "StoreType_d", "Assortment_a", "Assortment_b", \
	"Assortment_c", "PromoInterval_Feb,May,Aug,Nov", \
	"PromoInterval_Jan,Apr,Jul,Oct", "PromoInterval_Mar,Jun,Sept,Dec", \
	"Year", "Month", "Day", "WeekOfYear"]
params = {
	"objective": "reg:squarederror",
	"max_depth": 5,
    "nthread": 4
}
num_boost_round = 1300

dtrain = xgb.DMatrix(X_train[features], X_train.Sales)
dvalid = xgb.DMatrix(X_holdout[features], X_holdout.Sales)
watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=50, verbose_eval=True)

print("Validating")
yhat_holdout = gbm.predict(xgb.DMatrix(X_holdout[features]))
X_holdout["Prediction"] = yhat_holdout
yhat_train = gbm.predict(xgb.DMatrix(X_train[features]))
X_train["Prediction"] = yhat_train

print("Display the list of features by importance")
feature_importance = gbm.get_score(importance_type='gain')
feature_importance_sorted = sorted(feature_importance.items(), key=lambda kv: kv[1])
print(feature_importance_sorted)

# save model to file
print("Save the model to rossman.dat")
dump(gbm, "rossman.dat")

## Measure the error
print("")
print("#################################################")
train_pred = X_train["Prediction"].values
train_actuals = X_train["Sales"].values
rmspe_train = metric(train_pred, train_actuals)
print('RMSPE on TRAIN SET: {:.3f}'.format(rmspe_train))
predictions = X_holdout["Prediction"].values
actuals = X_holdout["Sales"].values
rmspe = metric(predictions, actuals)
print('RMSPE on TEST SET: {:.3f}'.format(rmspe))
print("#################################################")
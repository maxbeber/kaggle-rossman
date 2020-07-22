import numpy as np
import sys
import pandas as pd
import xgboost as xgb
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

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
    average_sales_by_store = df.groupby("Store")["Sales"].median()
    df["AverageSalesByStore"] = df.Store.map(average_sales_by_store)
    average_sales_by_day_of_week = df.groupby("DayOfWeek")["Sales"].median()
    df["AverageSalesByDayOfWeek"] = df.DayOfWeek.map(average_sales_by_store)
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
	print('RMSPE on TRAIN SET: {:.3f}'.format(rmspe_train))
	predictions = x_holdout["Prediction"].values
	actuals = y_holdout.values
	rmspe_test = rmspe(predictions, actuals)
	print('RMSPE on TEST SET: {:.3f}'.format(rmspe_test))
	print("#################################################")
	print("")

print("Loading training data.")
data = load_data()

print("Clean up data")
data = data_cleaning(data)

print("Feature engineering")
data = feature_engineering(data)

Y = data.Sales
X = data.drop(columns=["Sales"], axis=1)
X_train, X_holdout, Y_train, Y_holdout = train_test_split(X, Y, test_size=0.30, random_state=42, shuffle=True)

print("Train Random Forrest")
dtrain = xgb.DMatrix(X_train[features], Y_train)
dvalid = xgb.DMatrix(X_holdout[features], Y_holdout)
watchlist = [(dtrain, "train"), (dvalid, "eval")]
gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=50, verbose_eval=True)

print("Computing predictions")
yhat_holdout = gbm.predict(xgb.DMatrix(X_holdout[features]))
X_holdout["Prediction"] = yhat_holdout
yhat_train = gbm.predict(xgb.DMatrix(X_train[features]))
X_train["Prediction"] = yhat_train

print("Display the list of features by importance")
feature_score = gbm.get_score(importance_type="gain")
display_features_by_importance(feature_score)

# save model to file
print("Save the model to rossman.dat")
dump(gbm, "rossman.dat")

## Measure the error
display_metrics(X_train, Y_train, X_holdout, Y_holdout)
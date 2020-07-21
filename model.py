import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

def metric(preds, actuals):
	preds = preds.reshape(-1)
	actuals = actuals.reshape(-1)
	assert preds.shape == actuals.shape
	return 100 * np.linalg.norm((actuals - preds) / actuals) / np.sqrt(preds.shape[0])


def data_cleaning(df):
	df.loc[df.Open.isnull(), "Open"] = 1
	mean_imputer = SimpleImputer(strategy="median")
	df[["Customers"]] = mean_imputer.fit_transform(df[["Customers"]])
	df["DayOfWeek"] = df.Date.dt.dayofweek + 1
	categories = ["StateHoliday", "StoreType", "Assortment"]
	df[categories] = df[categories].fillna(value="Missing")
	numerical_features = ["Store", "Sales", "Open", "Promo", "SchoolHoliday", "CompetitionDistance", "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear", "Promo2", "Promo2SinceWeek", "Promo2SinceYear", "PromoInterval"]
	df[numerical_features] = df[numerical_features].fillna(value=-999)
	df["StateHoliday"] = df["StateHoliday"].replace({"0.0": "0"})
	integer_features = ["Promo", "Open", "Store", "Customers", "SchoolHoliday", "CompetitionDistance", "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear", "Promo2", "Promo2SinceWeek", "Promo2SinceYear"]
	df[integer_features]= df[integer_features].astype("int", copy=False)
	one_hot_encoding_features = ["StateHoliday", "StoreType", "Assortment", "PromoInterval"]
	df = pd.get_dummies(df, columns=one_hot_encoding_features)
	return df


def feature_engineering(df):
    df["Year"] = df.Date.dt.year
    df["Month"] = df.Date.dt.month
    df["Day"] = df.Date.dt.day
    df["WeekOfYear"] = df.Date.dt.weekofyear
    return df


print("Loading training data.")
train = pd.read_csv("data/train.csv", header=0, parse_dates=["Date"], dtype={"StateHoliday":object})
print(str(train.shape[0]) + " rows have been found.")

print("Loading store data.")
store = pd.read_csv("data/store.csv", header=0)
print(str(store.shape[0]) + " rows have been found.")

print("Loading test data")
test = pd.read_csv("data/test.csv", header=0, parse_dates=["Date"], dtype={"StateHoliday":object})
print(str(test.shape[0]) + " rows have been found.")

print("Filter rows where Sales is bigger than zero")
train = train[train.Sales > 0]
test = test[test.Sales > 0]

print("Filter rows where Open is true")
train = train[train.Open != 0]

print("Join train and store")
train = pd.merge(train, store, on="Store")
test = pd.merge(test, store, on="Store")

print("Clean up data")
train = data_cleaning(train)
#print(train.isnull().sum())

print("Feature engineering")
train = feature_engineering(train)
test = feature_engineering(test)
print(train.info())

print("Train Random Forrest")
features = ["Store", "Open", "DayOfWeek", "Year", "Month", "Day", "WeekOfYear", "Promo", "SchoolHoliday"]

naive_sales_per_store = train.groupby("Store")["Sales"].mean()
test["Prediction"] = test.Store.map(naive_sales_per_store)

# Train a Random Forest model
rf = RandomForestRegressor(n_jobs=4, verbose=True)
rf.fit(train[features], train.Sales)
test["Prediction"] = rf.predict(test[features])

# Measure the error
predictions = test["Prediction"].values
actuals = test["Sales"].values
rmspe = metric(predictions, actuals)
print('RMSPE: {:.3f}'.format(rmspe))

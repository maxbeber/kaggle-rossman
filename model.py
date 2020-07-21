import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

def metric(preds, actuals):
	preds = preds.reshape(-1)
	actuals = actuals.reshape(-1)
	assert preds.shape == actuals.shape
	return 100 * np.linalg.norm((actuals - preds) / actuals) / np.sqrt(preds.shape[0])


def data_cleaning(df): 
	df.loc[df.Open.isnull(), "Open"] = 1
	df.loc[df.Promo.isnull(), "Promo"] = -999
	df.Open = df.Open.astype('int64', copy=False)
	df.Promo = df.Promo.astype('int64', copy=False)
	df.Store = df.Store.astype('int64', copy=False)
	df.loc[df.CompetitionDistance.isnull(), "CompetitionDistance"] = -999
	df.loc[df.CompetitionOpenSinceMonth.isnull(), "CompetitionOpenSinceMonth"] = -999
	df.loc[df.CompetitionOpenSinceYear.isnull(), "CompetitionOpenSinceYear"] = -999
	df.loc[df.Promo2SinceYear.isnull(), "Promo2SinceYear"] = -999
	df.loc[df.Promo2SinceWeek.isnull(), "Promo2SinceWeek"] = -999
	df.loc[df.SchoolHoliday.isnull(), "SchoolHoliday"] = -999
	mean_imputer = SimpleImputer(strategy="median")
	df[["Customers"]] = mean_imputer.fit_transform(df[["Customers"]])
	df.loc[:, "DayOfWeek"] = df.Date.dt.dayofweek + 1
	df.DayOfWeek = df.DayOfWeek.astype('int64', copy=False)
	df.Promo2SinceWeek = df.Promo2SinceWeek.astype('int64', copy=False)
	df.Promo2SinceYear = df.Promo2SinceYear.astype('int64', copy=False)
	df.CompetitionOpenSinceMonth = df.CompetitionOpenSinceMonth.astype('int64', copy=False)
	df.CompetitionOpenSinceYear = df.CompetitionOpenSinceYear.astype('int64', copy=False)
	df.Customers = df.Customers.astype('int64', copy=False)
	df.CompetitionDistance = df.CompetitionDistance.astype('int64', copy=False)
	categories = ["StateHoliday", "StoreType", "Assortment"]
	df.loc[:, categories].fillna(value="Missing", inplace=True)
	df.loc[:, "StateHoliday"] = df.StateHoliday.replace({"0.0": "0"})
	integer_features = ["Open", "Store"]
	df[integer_features].astype("int", copy=False)
	one_hot_encoding_features = ["SchoolHoliday", "StateHoliday", "StoreType", "Assortment", "PromoInterval"]
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

print("Filter rows where Sales is bigger than zero")
train = train[train.Sales > 0]

print("Filter rows where Open is true")
train = train[train.Open != 0]

print("Join train and store")
train = pd.merge(train, store, on="Store")

df_train, df_test = train_test_split(train, test_size=0.25, random_state=42, shuffle=True)

print("Clean up data")
df_train = data_cleaning(df_train)
df_test = data_cleaning(df_test)
#print(df_train.isnull().sum())

print("Feature engineering")
df_train = feature_engineering(df_train)
df_test = feature_engineering(df_test)
#print(df_train.info())
#print(df_test.info())

print("Train Random Forrest")
features = ["Store", "Customers", "CompetitionDistance", "StoreType_d", "DayOfWeek", "Promo", "Open",
"Assortment_b", "Promo2SinceWeek", "StoreType_b", "WeekOfYear", "Promo2SinceYear", "Month", "Assortment_a"]

rf = RandomForestRegressor(n_jobs=4, verbose=True)
rf.fit(df_train[features], df_train.Sales)
df_test["Prediction"] = rf.predict(df_test[features])

## Feature Importance:
importances_rf = pd.Series(rf.feature_importances_, index = df_train[features].columns)
sorted_importances_rf = importances_rf.sort_values(ascending=False).nlargest(10)
print(sorted_importances_rf)

# Measure the error
predictions = df_test["Prediction"].values
actuals = df_test["Sales"].values
rmspe = metric(predictions, actuals)
print('RMSPE: {:.3f}'.format(rmspe))

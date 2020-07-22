import pandas as pd
from sklearn.impute import SimpleImputer


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


def m_group(x, type, quantiles):
    if x <= quantiles[type].iloc[0]:
        return 4
    elif x <= quantiles[type].iloc[1]:
        return 3
    elif x <= quantiles[type].iloc[2]:
        return 2
    else:
        return 1
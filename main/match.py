#Import e funzioni utili
import pandas as pd
import os
import numpy as np

def load_data(filepath, del_time = False, del_spread = False, start_month = False):

    # import data
    df = pd.read_csv(filepath)
    df.fillna(0, inplace = True)

    # delete first two column and empty LOB
    df.drop(columns = ["Unnamed: 0", "key"], inplace = True)
    df.drop(df[df["AskPrice_0"] == 0].index.to_list(), inplace = True)
    df.drop(df[df["BidPrice_0"] == 0].index.to_list(), inplace = True)

    # scale price to € cent and add mid price and spread
    df.iloc[:,1:41:4] = df.iloc[:,1:41:4] * 100
    df.iloc[:,3:43:4] = df.iloc[:,3:43:4] * 100
    df["MidPrice"] = (df["BidPrice_0"] + df["AskPrice_0"]) / 2
    df["Spread"] = df["AskPrice_0"] - df["BidPrice_0"]

    # transform the column Datetime from string to datetime
    df["Datetime"]= pd.to_datetime(df["Datetime"])

    # create a new column that represent second to start of the month if start_month
    # is True otherwise create a new column that represent second to midnight
    seconds = np.zeros(len(df))

    if start_month is True:
        for i, date in enumerate(df["Datetime"]):
            seconds[i] = date.second + 60 * date.minute + 3600 * date.hour + \
                         date.microsecond * 1e-6 + (date.day - 1) * 24 * 3600
    else:
        for i, date in enumerate(df["Datetime"]):
            seconds[i] = date.second + 60 * date.minute + 3600 * date.hour + \
                         date.microsecond * 1e-6

    df["Seconds"] = seconds

    # delete first and last hour of trading
    if del_time is True:
        df = df.loc[df["Datetime"].dt.hour > 6]
        df = df.loc[df["Datetime"].dt.hour < 16]

    # delete spread < 0
    if del_spread is True:
        df = df.loc[df["Spread"] > 0]

    df.reset_index(inplace = True, drop = True)

    return df

def clean_data(clean_data):

    df = clean_data.iloc[:,1:41].diff().fillna(0)

    df["Price"] = df["AskVolume_0"]*0
    df["Volume"] = df["AskVolume_0"]*0
    df["Sign"] = df["AskVolume_0"]*0
    df["Quote"] = df["AskVolume_0"]*0
    df["Quote"].replace(0, "NoBest", inplace = True)
    df["Type"] = df["BidPrice_0"]*0

    #infer sign price volume quote and type from the LOB dataframe

    # BidPrice_0 > 0
    con = (df["BidPrice_0"] > 0)
    a = df.loc[con].index.to_numpy()
    df.loc[a,["Type"]] = "Limit"
    df.loc[a,["Quote"]] = 0
    df.loc[a,["Sign"]] = 1
    df.loc[a,["Price"]] = clean_data.loc[a]["BidPrice_0"]
    df.loc[a,["Volume"]] = clean_data.loc[a]["BidVolume_0"]

    # BidPrice_0 < 0
    con = (df["BidPrice_0"] < 0)
    a = df.loc[con].index.to_numpy()
    df.loc[a,["Type"]] = "Market/Cancel"
    df.loc[a,["Quote"]] = 0
    df.loc[a,["Sign"]] = 1
    df.loc[a,["Price"]] = clean_data.loc[a-1]["BidPrice_0"].to_list()
    df.loc[a,["Volume"]] = clean_data.loc[a-1]["BidVolume_0"].to_list()

    # AskPrice_0 > 0
    con = (df["AskPrice_0"] > 0)
    a = df.loc[con].index.to_numpy()
    df.loc[a,["Type"]] = "Market/Cancel"
    df.loc[a,["Quote"]] = 0
    df.loc[a,["Sign"]] = -1
    df.loc[a,["Price"]] = clean_data.loc[a-1]["AskPrice_0"].to_list()
    df.loc[a,["Volume"]] = clean_data.loc[a-1]["AskVolume_0"].to_list()

    # AskPrice_0 < 0
    con = (df["AskPrice_0"] < 0)
    a = df.loc[con].index.to_numpy()
    df.loc[a,["Type"]] = "Limit"
    df.loc[a,["Quote"]] = 0
    df.loc[a,["Sign"]] = -1
    df.loc[a,["Price"]] = clean_data.loc[a]["AskPrice_0"]
    df.loc[a,["Volume"]] = clean_data.loc[a]["AskVolume_0"]

    # BidVolume_0 > 0
    con = (df["BidPrice_0"] == 0) & (df["BidVolume_0"] > 0)
    a = df.loc[con].index.to_numpy()
    df.loc[a,["Type"]] = "Limit"
    df.loc[a,["Quote"]] = 0
    df.loc[a,["Sign"]] = 1
    df.loc[a,["Price"]] = clean_data.loc[a]["BidPrice_0"]
    df.loc[a,["Volume"]] = df.loc[a]["BidVolume_0"]

    # BidVolume_0 < 0
    con = (df["BidPrice_0"] == 0) & (df["BidVolume_0"] < 0)
    a = df.loc[con].index.to_numpy()
    df.loc[a,["Type"]] = "Market/Cancel"
    df.loc[a,["Quote"]] = 0
    df.loc[a,["Sign"]] = 1
    df.loc[a,["Price"]] = clean_data.loc[a]["BidPrice_0"]
    df.loc[a,["Volume"]] = df.loc[a]["BidVolume_0"] * (-1)

    # AskVolume_0 > 0
    con = (df["AskPrice_0"] == 0) & (df["AskVolume_0"] > 0)
    a = df.loc[con].index.to_numpy()
    df.loc[a,["Type"]] = "Limit"
    df.loc[a,["Quote"]] = 0
    df.loc[a,["Sign"]] = -1
    df.loc[a,["Price"]] = clean_data.loc[a]["AskPrice_0"]
    df.loc[a,["Volume"]] = df.loc[a]["AskVolume_0"]

    # AskVolume_0 < 0
    con = (df["BidPrice_0"] == 0) & (df["AskVolume_0"] < 0)
    a = df.loc[con].index.to_numpy()
    df.loc[a,["Type"]] = "Market/Cancel"
    df.loc[a,["Quote"]] = 0
    df.loc[a,["Sign"]] = -1
    df.loc[a,["Price"]] = clean_data.loc[a]["AskPrice_0"]
    df.loc[a,["Volume"]] = df.loc[a]["AskVolume_0"]* (-1)

    # BidPrice_1 > 0
    con = (df["BidPrice_0"] == 0) & (df["BidVolume_0"] == 0) & (df["BidPrice_1"] > 0)
    a = df.loc[con].index.to_numpy()
    df.loc[a,["Type"]] = "Limit"
    df.loc[a,["Quote"]] = 1
    df.loc[a,["Sign"]] = 1
    df.loc[a,["Price"]] = clean_data.loc[a]["BidPrice_1"]
    df.loc[a,["Volume"]] = clean_data.loc[a]["BidVolume_1"]

    # BidPrice_1 < 0
    con = (df["BidPrice_0"] == 0) & (df["BidVolume_0"] == 0) & (df["BidPrice_1"] < 0)
    a = df.loc[con].index.to_numpy()
    df.loc[a,["Type"]] = "Market/Cancel"
    df.loc[a,["Quote"]] = 1
    df.loc[a,["Sign"]] = 1
    df.loc[a,["Price"]] = clean_data.loc[a-1]["BidPrice_1"].to_list()
    df.loc[a,["Volume"]] = clean_data.loc[a-1]["BidVolume_1"].to_list()

    # AskPrice_1 > 0
    con = (df["AskPrice_0"] == 0) & (df["AskVolume_0"] == 0) & (df["AskPrice_1"] > 0)
    a = df.loc[con].index.to_numpy()
    df.loc[a,["Type"]] = "Market/Cancel"
    df.loc[a,["Quote"]] = 1
    df.loc[a,["Sign"]] = -1
    df.loc[a,["Price"]] = clean_data.loc[a-1]["AskPrice_1"].to_list()
    df.loc[a,["Volume"]] = clean_data.loc[a-1]["AskVolume_1"].to_list()

    # AskPrice_1 < 0
    con = (df["AskPrice_0"] == 0) & (df["AskVolume_0"] == 0) & (df["AskPrice_1"] < 0)
    a = df.loc[con].index.to_numpy()
    df.loc[a,["Type"]] = "Limit"
    df.loc[a,["Quote"]] = 1
    df.loc[a,["Sign"]] = -1
    df.loc[a,["Price"]] = clean_data.loc[a]["AskPrice_1"]
    df.loc[a,["Volume"]] = clean_data.loc[a]["AskVolume_1"]

    # BidVolume_1 > 0
    con = (df["BidPrice_0"] == 0) & (df["BidVolume_0"] == 0) & (df["BidPrice_1"] == 0) & \
          (df["BidVolume_1"] >0)

    a = df.loc[con].index.to_numpy()
    df.loc[a,["Type"]] = "Limit"
    df.loc[a,["Quote"]] = 1
    df.loc[a,["Sign"]] = 1
    df.loc[a,["Price"]] = clean_data.loc[a]["BidPrice_1"]
    df.loc[a,["Volume"]] = df.loc[a]["BidVolume_1"]

    # BidVolume_1 < 0
    con = (df["BidPrice_0"] == 0) & (df["BidVolume_0"] == 0) & (df["BidPrice_1"] == 0) &\
          (df["BidVolume_1"] < 0)

    a = df.loc[con].index.to_numpy()
    df.loc[a,["Type"]] = "Market/Cancel"
    df.loc[a,["Quote"]] = 1
    df.loc[a,["Sign"]] = 1
    df.loc[a,["Price"]] = clean_data.loc[a]["BidPrice_1"]
    df.loc[a,["Volume"]] = df.loc[a]["BidVolume_1"] * (-1)

    # AskVolume_1 > 0
    con = (df["AskPrice_0"] == 0) & (df["AskVolume_0"] == 0) & (df["AskPrice_1"] == 0) &\
          (df["AskVolume_1"] > 0)

    a = df.loc[con].index.to_numpy()
    df.loc[a,["Type"]] = "Limit"
    df.loc[a,["Quote"]] = 1
    df.loc[a,["Sign"]] = -1
    df.loc[a,["Price"]] = clean_data.loc[a]["AskPrice_1"]
    df.loc[a,["Volume"]] = df.loc[a]["AskVolume_1"]

    # AskVolume_1 < 0
    con = (df["AskPrice_0"] == 0) & (df["AskVolume_0"] == 0) & (df["AskPrice_1"] == 0) &\
          (df["AskVolume_1"] < 0)

    a = df.loc[con].index.to_numpy()
    df.loc[a,["Type"]] = "Market/Cancel"
    df.loc[a,["Quote"]] = 1
    df.loc[a,["Sign"]] = -1
    df.loc[a,["Price"]] = clean_data.loc[a]["AskPrice_1"]
    df.loc[a,["Volume"]] = df.loc[a]["AskVolume_1"]* (-1)

    # BidPrice_2 > 0
    con = (df["BidPrice_0"] == 0) & (df["BidVolume_0"] == 0) & (df["BidPrice_1"] == 0) & \
          (df["BidVolume_1"] == 0) & (df["BidPrice_2"] > 0)

    a = df.loc[con].index.to_numpy()
    df.loc[a,["Type"]] = "Limit"
    df.loc[a,["Quote"]] = 2
    df.loc[a,["Sign"]] = 1
    df.loc[a,["Price"]] = clean_data.loc[a]["BidPrice_2"]
    df.loc[a,["Volume"]] = clean_data.loc[a]["BidVolume_2"]

    # BidPrice_2 < 0
    con = (df["BidPrice_0"] == 0) & (df["BidVolume_0"] == 0) & (df["BidPrice_1"] == 0) & \
          (df["BidVolume_1"] == 0) & (df["BidPrice_2"] < 0)

    a = df.loc[con].index.to_numpy()
    df.loc[a,["Type"]] = "Market/Cancel"
    df.loc[a,["Quote"]] = 2
    df.loc[a,["Sign"]] = 1
    df.loc[a,["Price"]] = clean_data.loc[a-1]["BidPrice_2"].to_list()
    df.loc[a,["Volume"]] = clean_data.loc[a-1]["BidVolume_2"].to_list()

    # AskPrice_2 > 0
    con = (df["AskPrice_0"] == 0) & (df["AskVolume_0"] == 0) & (df["AskPrice_1"] == 0) & \
          (df["AskVolume_1"] == 0) & (df["AskPrice_2"] > 0)

    a = df.loc[con].index.to_numpy()
    df.loc[a,["Type"]] = "Market/Cancel"
    df.loc[a,["Quote"]] = 2
    df.loc[a,["Sign"]] = -1
    df.loc[a,["Price"]] = clean_data.loc[a-1]["AskPrice_2"].to_list()
    df.loc[a,["Volume"]] = clean_data.loc[a-1]["AskVolume_2"].to_list()

    # AskPrice_2 < 0
    con = (df["AskPrice_0"] == 0) & (df["AskVolume_0"] == 0) & (df["AskPrice_1"] == 0) & \
          (df["AskVolume_1"] == 0) & (df["AskPrice_2"] < 0)

    a = df.loc[con].index.to_numpy()
    df.loc[a,["Type"]] = "Limit"
    df.loc[a,["Quote"]] = 2
    df.loc[a,["Sign"]] = -1
    df.loc[a,["Price"]] = clean_data.loc[a]["AskPrice_2"]
    df.loc[a,["Volume"]] = clean_data.loc[a]["AskVolume_2"]


    # BidVolume_2 > 0
    con = (df["BidPrice_0"] == 0) & (df["BidVolume_0"] == 0) & (df["BidPrice_1"] == 0) & \
          (df["BidVolume_1"] == 0) & (df["BidPrice_2"] == 0) & (df["BidVolume_2"] > 0)

    a = df.loc[con].index.to_numpy()
    df.loc[a,["Type"]] = "Limit"
    df.loc[a,["Quote"]] = 2
    df.loc[a,["Sign"]] = 1
    df.loc[a,["Price"]] = clean_data.loc[a]["BidPrice_2"]
    df.loc[a,["Volume"]] = df.loc[a]["BidVolume_2"]

    # BidVolume_2 < 0
    con = (df["BidPrice_0"] == 0) & (df["BidVolume_0"] == 0) & (df["BidPrice_1"] == 0) & \
          (df["BidVolume_1"] == 0) & (df["BidPrice_2"] == 0) & (df["BidVolume_2"] < 0)

    a = df.loc[con].index.to_numpy()
    df.loc[a,["Type"]] = "Market/Cancel"
    df.loc[a,["Quote"]] = 2
    df.loc[a,["Sign"]] = 1
    df.loc[a,["Price"]] = clean_data.loc[a]["BidPrice_2"]
    df.loc[a,["Volume"]] = df.loc[a]["BidVolume_2"] * (-1)

    # AskVolume_2 > 0
    con = (df["AskPrice_0"] == 0) & (df["AskVolume_0"] == 0) & (df["AskPrice_1"] == 0) & \
          (df["AskVolume_1"] == 0) & (df["AskPrice_2"] == 0) & (df["AskVolume_2"] > 0)
    a = df.loc[con].index.to_numpy()
    df.loc[a,["Type"]] = "Limit"
    df.loc[a,["Quote"]] = 2
    df.loc[a,["Sign"]] = -1
    df.loc[a,["Price"]] = clean_data.loc[a]["AskPrice_2"]
    df.loc[a,["Volume"]] = df.loc[a]["AskVolume_2"]

    # AskVolume_2 < 0
    con = (df["AskPrice_0"] == 0) & (df["AskVolume_0"] == 0) & (df["AskPrice_1"] == 0) & \
          (df["AskVolume_1"] == 0) & (df["AskPrice_2"] == 0) & (df["AskVolume_2"] < 0)
    a = df.loc[con].index.to_numpy()
    df.loc[a,["Type"]] = "Market/Cancel"
    df.loc[a,["Quote"]] = 2
    df.loc[a,["Sign"]] = -1
    df.loc[a,["Price"]] = clean_data.loc[a]["AskPrice_2"]
    df.loc[a,["Volume"]] = df.loc[a]["AskVolume_2"]* (-1)


    # BidPrice_3 > 0
    con = (df["BidPrice_0"] == 0) & (df["BidVolume_0"] == 0) & (df["BidPrice_1"] == 0) & \
          (df["BidVolume_1"] == 0) & (df["BidPrice_2"] == 0) & (df["BidVolume_2"] == 0) & \
          (df["BidPrice_3"] > 0)

    a = df.loc[con].index.to_numpy()
    df.loc[a,["Type"]] = "Limit"
    df.loc[a,["Quote"]] = 3
    df.loc[a,["Sign"]] = 1
    df.loc[a,["Price"]] = clean_data.loc[a]["BidPrice_3"]
    df.loc[a,["Volume"]] = clean_data.loc[a]["BidVolume_3"]

    # BidPrice_3 < 0
    con = (df["BidPrice_0"] == 0) & (df["BidVolume_0"] == 0) & (df["BidPrice_1"] == 0) & \
          (df["BidVolume_1"] == 0) & (df["BidPrice_2"] == 0) & (df["BidVolume_2"] == 0) & \
          (df["BidPrice_3"] < 0)

    a = df.loc[con].index.to_numpy()
    df.loc[a,["Type"]] = "Market/Cancel"
    df.loc[a,["Quote"]] = 3
    df.loc[a,["Sign"]] = 1
    df.loc[a,["Price"]] = clean_data.loc[a-1]["BidPrice_3"].to_list()
    df.loc[a,["Volume"]] = clean_data.loc[a-1]["BidVolume_3"].to_list()

    # AskPrice_3 > 0
    con = (df["AskPrice_0"] == 0) & (df["AskVolume_0"] == 0) & (df["AskPrice_1"] == 0) & \
          (df["AskVolume_1"] == 0) & (df["AskPrice_2"] == 0) & (df["AskVolume_2"] == 0) & \
          (df["AskPrice_3"]  > 0)

    a = df.loc[con].index.to_numpy()
    df.loc[a,["Type"]] = "Market/Cancel"
    df.loc[a,["Quote"]] = 3
    df.loc[a,["Sign"]] = -1
    df.loc[a,["Price"]] = clean_data.loc[a-1]["AskPrice_3"].to_list()
    df.loc[a,["Volume"]] = clean_data.loc[a-1]["AskVolume_3"].to_list()

    # AskPrice_3 < 0
    con = (df["AskPrice_0"] == 0) & (df["AskVolume_0"] == 0) & (df["AskPrice_1"] == 0) & \
          (df["AskVolume_1"] == 0) & (df["AskPrice_2"] == 0) & (df["AskVolume_2"] == 0) & \
          (df["AskPrice_3"]  < 0)

    a = df.loc[con].index.to_numpy()
    df.loc[a,["Type"]] = "Limit"
    df.loc[a,["Quote"]] = 3
    df.loc[a,["Sign"]] = -1
    df.loc[a,["Price"]] = clean_data.loc[a]["AskPrice_3"]
    df.loc[a,["Volume"]] = clean_data.loc[a]["AskVolume_3"]


    # BidVolume_3 > 0
    con = (df["BidPrice_0"] == 0) & (df["BidVolume_0"] == 0) & (df["BidPrice_1"] == 0) & \
          (df["BidVolume_1"] == 0) & (df["BidPrice_2"] == 0) & (df["BidVolume_2"] == 0) & \
          (df["BidPrice_3"] == 0) & (df["BidVolume_3"] > 0)

    a = df.loc[con].index.to_numpy()
    df.loc[a,["Type"]] = "Limit"
    df.loc[a,["Quote"]] = 3
    df.loc[a,["Sign"]] = 1
    df.loc[a,["Price"]] = clean_data.loc[a]["BidPrice_3"]
    df.loc[a,["Volume"]] = df.loc[a]["BidVolume_3"]

    # BidVolume_3 < 0
    con = (df["BidPrice_0"] == 0) & (df["BidVolume_0"] == 0) & (df["BidPrice_1"] == 0) & \
          (df["BidVolume_1"] == 0) & (df["BidPrice_2"] == 0) & (df["BidVolume_2"] == 0) & \
          (df["BidPrice_3"] == 0) & (df["BidVolume_3"] < 0)

    a = df.loc[con].index.to_numpy()
    df.loc[a,["Type"]] = "Market/Cancel"
    df.loc[a,["Quote"]] = 3
    df.loc[a,["Sign"]] = 1
    df.loc[a,["Price"]] = clean_data.loc[a]["BidPrice_3"]
    df.loc[a,["Volume"]] = df.loc[a]["BidVolume_3"] * (-1)

    # AskVolume_3 > 0
    con = (df["AskPrice_0"] == 0) & (df["AskVolume_0"] == 0) & (df["AskPrice_1"] == 0) & \
          (df["AskVolume_1"] == 0) & (df["AskPrice_2"] == 0) & (df["AskVolume_2"] == 0) & \
          (df["AskPrice_3"]  == 0) & (df["AskVolume_3"] > 0)

    a = df.loc[con].index.to_numpy()
    df.loc[a,["Type"]] = "Limit"
    df.loc[a,["Quote"]] = 3
    df.loc[a,["Sign"]] = -1
    df.loc[a,["Price"]] = clean_data.loc[a]["AskPrice_3"]
    df.loc[a,["Volume"]] = df.loc[a]["AskVolume_3"]

    # AskVolume_3 < 0
    con = (df["AskPrice_0"] == 0) & (df["AskVolume_0"] == 0) & (df["AskPrice_1"] == 0) & \
          (df["AskVolume_1"] == 0) & (df["AskPrice_2"] == 0) & (df["AskVolume_2"] == 0) & \
          (df["AskPrice_3"]  == 0) & (df["AskVolume_3"] < 0)

    a = df.loc[con].index.to_numpy()
    df.loc[a,["Type"]] = "Market/Cancel"
    df.loc[a,["Quote"]] = 3
    df.loc[a,["Sign"]] = -1
    df.loc[a,["Price"]] = clean_data.loc[a]["AskPrice_3"]
    df.loc[a,["Volume"]] = df.loc[a]["AskVolume_3"]* (-1)

    # BidPrice_4 > 0
    con = (df["BidPrice_0"] == 0) & (df["BidVolume_0"] == 0) & (df["BidPrice_1"] == 0) & \
          (df["BidVolume_1"] == 0) & (df["BidPrice_2"] == 0) & (df["BidVolume_2"] == 0) & \
          (df["BidPrice_3"] == 0) & (df["BidVolume_3"] == 0) & (df["BidPrice_4"] > 0)

    a = df.loc[con].index.to_numpy()
    df.loc[a,["Type"]] = "Limit"
    df.loc[a,["Quote"]] = 4
    df.loc[a,["Sign"]] = 1
    df.loc[a,["Price"]] = clean_data.loc[a]["BidPrice_4"]
    df.loc[a,["Volume"]] = clean_data.loc[a]["BidVolume_4"]

    # BidPrice_4 < 0
    con = (df["BidPrice_0"] == 0) & (df["BidVolume_0"] == 0) & (df["BidPrice_1"] == 0) & \
          (df["BidVolume_1"] == 0) & (df["BidPrice_2"] == 0) & (df["BidVolume_2"] == 0) & \
          (df["BidPrice_3"] == 0) & (df["BidVolume_3"] == 0) & (df["BidPrice_4"] < 0)

    a = df.loc[con].index.to_numpy()
    df.loc[a,["Type"]] = "Market/Cancel"
    df.loc[a,["Quote"]] = 4
    df.loc[a,["Sign"]] = 1
    df.loc[a,["Price"]] = clean_data.loc[a-1]["BidPrice_4"].to_list()
    df.loc[a,["Volume"]] = clean_data.loc[a-1]["BidVolume_4"].to_list()

    # AskPrice_4 > 0
    con = (df["AskPrice_0"] == 0) & (df["AskVolume_0"] == 0) & (df["AskPrice_1"] == 0) & \
          (df["AskVolume_1"] == 0) & (df["AskPrice_2"] == 0) & (df["AskVolume_2"] == 0) & \
          (df["AskPrice_3"]  == 0) & (df["AskVolume_3"] == 0) & (df["AskPrice_4"] > 0)

    a = df.loc[con].index.to_numpy()
    df.loc[a,["Type"]] = "Market/Cancel"
    df.loc[a,["Quote"]] = 4
    df.loc[a,["Sign"]] = -1
    df.loc[a,["Price"]] = clean_data.loc[a-1]["AskPrice_4"].to_list()
    df.loc[a,["Volume"]] = clean_data.loc[a-1]["AskVolume_4"].to_list()

    # AskPrice_4 < 0
    con = (df["AskPrice_0"] == 0) & (df["AskVolume_0"] == 0) & (df["AskPrice_1"] == 0) & \
          (df["AskVolume_1"] == 0) & (df["AskPrice_2"] == 0) & (df["AskVolume_2"] == 0) & \
          (df["AskPrice_3"]  == 0) & (df["AskVolume_3"] == 0) & (df["AskPrice_4"] < 0)

    a = df.loc[con].index.to_numpy()
    df.loc[a,["Type"]] = "Limit"
    df.loc[a,["Quote"]] = 4
    df.loc[a,["Sign"]] = -1
    df.loc[a,["Price"]] = clean_data.loc[a]["AskPrice_4"]
    df.loc[a,["Volume"]] = clean_data.loc[a]["AskVolume_4"]


    # BidVolume_4 > 0
    con = (df["BidPrice_0"] == 0) & (df["BidVolume_0"] == 0) & (df["BidPrice_1"] == 0) & \
          (df["BidVolume_1"] == 0) & (df["BidPrice_2"] == 0) & (df["BidVolume_2"] == 0) & \
          (df["BidPrice_3"] == 0) & (df["BidVolume_3"] == 0) & (df["BidPrice_4"] == 0) & \
          (df["BidVolume_4"] > 0)

    a = df.loc[con].index.to_numpy()
    df.loc[a,["Type"]] = "Limit"
    df.loc[a,["Quote"]] = 4
    df.loc[a,["Sign"]] = 1
    df.loc[a,["Price"]] = clean_data.loc[a]["BidPrice_4"]
    df.loc[a,["Volume"]] = df.loc[a]["BidVolume_4"]

    # BidVolume_4 < 0
    con = (df["BidPrice_0"] == 0) & (df["BidVolume_0"] == 0) & (df["BidPrice_1"] == 0) & \
          (df["BidVolume_1"] == 0) & (df["BidPrice_2"] == 0) & (df["BidVolume_2"] == 0) & \
          (df["BidPrice_3"] == 0) & (df["BidVolume_3"] == 0) & (df["BidPrice_4"] == 0) & \
          (df["BidVolume_4"] < 0)

    a = df.loc[con].index.to_numpy()
    df.loc[a,["Type"]] = "Market/Cancel"
    df.loc[a,["Quote"]] = 4
    df.loc[a,["Sign"]] = 1
    df.loc[a,["Price"]] = clean_data.loc[a]["BidPrice_4"]
    df.loc[a,["Volume"]] = df.loc[a]["BidVolume_4"] * (-1)

    # AskVolume_4 > 0
    con = (df["AskPrice_0"] == 0) & (df["AskVolume_0"] == 0) & (df["AskPrice_1"] == 0) & \
          (df["AskVolume_1"] == 0) & (df["AskPrice_2"] == 0) & (df["AskVolume_2"] == 0) & \
          (df["AskPrice_3"]  == 0) & (df["AskVolume_3"] == 0) & (df["AskPrice_4"] == 0) & \
          (df["AskVolume_4"] > 0)

    a = df.loc[con].index.to_numpy()
    df.loc[a,["Type"]] = "Limit"
    df.loc[a,["Quote"]] = 4
    df.loc[a,["Sign"]] = -1
    df.loc[a,["Price"]] = clean_data.loc[a]["AskPrice_4"]
    df.loc[a,["Volume"]] = df.loc[a]["AskVolume_4"]

    # AskVolume_4 < 0
    con = (df["AskPrice_0"] == 0) & (df["AskVolume_0"] == 0) & (df["AskPrice_1"] == 0) & \
          (df["AskVolume_1"] == 0) & (df["AskPrice_2"] == 0) & (df["AskVolume_2"] == 0) & \
          (df["AskPrice_3"]  == 0) & (df["AskVolume_3"] == 0) & (df["AskPrice_4"] == 0) & \
          (df["AskVolume_4"] < 0)

    a = df.loc[con].index.to_numpy()
    df.loc[a,["Type"]] = "Market/Cancel"
    df.loc[a,["Quote"]] = 4
    df.loc[a,["Sign"]] = -1
    df.loc[a,["Price"]] = clean_data.loc[a]["AskPrice_4"]
    df.loc[a,["Volume"]] = df.loc[a]["AskVolume_4"]* (-1)

    # BidPrice_5 > 0
    con = (df["BidPrice_0"] == 0) & (df["BidVolume_0"] == 0) & (df["BidPrice_1"] == 0) & \
          (df["BidVolume_1"] == 0) & (df["BidPrice_2"] == 0) & (df["BidVolume_2"] == 0) & \
          (df["BidPrice_3"] == 0) & (df["BidVolume_3"] == 0) & (df["BidPrice_4"] == 0) & \
          (df["BidVolume_4"] == 0) & (df["BidPrice_5"] > 0)

    a = df.loc[con].index.to_numpy()
    df.loc[a,["Type"]] = "Limit"
    df.loc[a,["Quote"]] = 5
    df.loc[a,["Sign"]] = 1
    df.loc[a,["Price"]] = clean_data.loc[a]["BidPrice_5"]
    df.loc[a,["Volume"]] = clean_data.loc[a]["BidVolume_5"]

    # BidPrice_5 < 0
    con = (df["BidPrice_0"] == 0) & (df["BidVolume_0"] == 0) & (df["BidPrice_1"] == 0) & \
          (df["BidVolume_1"] == 0) & (df["BidPrice_2"] == 0) & (df["BidVolume_2"] == 0) & \
          (df["BidPrice_3"] == 0) & (df["BidVolume_3"] == 0) & (df["BidPrice_4"] == 0) & \
          (df["BidVolume_4"] == 0) & (df["BidPrice_5"] < 0)

    a = df.loc[con].index.to_numpy()
    df.loc[a,["Type"]] = "Market/Cancel"
    df.loc[a,["Quote"]] = 5
    df.loc[a,["Sign"]] = 1
    df.loc[a,["Price"]] = clean_data.loc[a-1]["BidPrice_5"].to_list()
    df.loc[a,["Volume"]] = clean_data.loc[a-1]["BidVolume_5"].to_list()

    # AskPrice_5 > 0
    con = (df["AskPrice_0"] == 0) & (df["AskVolume_0"] == 0) & (df["AskPrice_1"] == 0) & \
          (df["AskVolume_1"] == 0) & (df["AskPrice_2"] == 0) & (df["AskVolume_2"] == 0) & \
          (df["AskPrice_3"]  == 0) & (df["AskVolume_3"] == 0) & (df["AskPrice_4"] == 0) & \
          (df["AskVolume_4"] == 0) & (df["AskPrice_5"] > 0)

    a = df.loc[con].index.to_numpy()
    df.loc[a,["Type"]] = "Market/Cancel"
    df.loc[a,["Quote"]] = 5
    df.loc[a,["Sign"]] = -1
    df.loc[a,["Price"]] = clean_data.loc[a-1]["AskPrice_5"].to_list()
    df.loc[a,["Volume"]] = clean_data.loc[a-1]["AskVolume_5"].to_list()

    # AskPrice_5 < 0
    con = (df["AskPrice_0"] == 0) & (df["AskVolume_0"] == 0) & (df["AskPrice_1"] == 0) & \
          (df["AskVolume_1"] == 0) & (df["AskPrice_2"] == 0) & (df["AskVolume_2"] == 0) & \
          (df["AskPrice_3"]  == 0) & (df["AskVolume_3"] == 0) & (df["AskPrice_4"] == 0) & \
          (df["AskVolume_4"] == 0) & (df["AskPrice_5"] < 0)

    a = df.loc[con].index.to_numpy()
    df.loc[a,["Type"]] = "Limit"
    df.loc[a,["Quote"]] = 5
    df.loc[a,["Sign"]] = -1
    df.loc[a,["Price"]] = clean_data.loc[a]["AskPrice_5"]
    df.loc[a,["Volume"]] = clean_data.loc[a]["AskVolume_5"]


    # BidVolume_5 > 0
    con = (df["BidPrice_0"] == 0) & (df["BidVolume_0"] == 0) & (df["BidPrice_1"] == 0) & \
          (df["BidVolume_1"] == 0) & (df["BidPrice_2"] == 0) & (df["BidVolume_2"] == 0) & \
          (df["BidPrice_3"] == 0) & (df["BidVolume_3"] == 0) & (df["BidPrice_4"] == 0) & \
          (df["BidVolume_4"] == 0) & (df["BidPrice_5"] == 0) & (df["BidVolume_5"] > 0)

    a = df.loc[con].index.to_numpy()
    df.loc[a,["Type"]] = "Limit"
    df.loc[a,["Quote"]] = 5
    df.loc[a,["Sign"]] = 1
    df.loc[a,["Price"]] = clean_data.loc[a]["BidPrice_5"]
    df.loc[a,["Volume"]] = df.loc[a]["BidVolume_5"]

    # BidVolume_5 < 0
    con = (df["BidPrice_0"] == 0) & (df["BidVolume_0"] == 0) & (df["BidPrice_1"] == 0) & \
          (df["BidVolume_1"] == 0) & (df["BidPrice_2"] == 0) & (df["BidVolume_2"] == 0) & \
          (df["BidPrice_3"] == 0) & (df["BidVolume_3"] == 0) & (df["BidPrice_4"] == 0) & \
          (df["BidVolume_4"] == 0) & (df["BidPrice_5"] == 0) & (df["BidVolume_5"] < 0)

    a = df.loc[con].index.to_numpy()
    df.loc[a,["Type"]] = "Market/Cancel"
    df.loc[a,["Quote"]] = 5
    df.loc[a,["Sign"]] = 1
    df.loc[a,["Price"]] = clean_data.loc[a]["BidPrice_5"]
    df.loc[a,["Volume"]] = df.loc[a]["BidVolume_5"] * (-1)

    # AskVolume_5 > 0
    con = (df["AskPrice_0"] == 0) & (df["AskVolume_0"] == 0) & (df["AskPrice_1"] == 0) & \
          (df["AskVolume_1"] == 0) & (df["AskPrice_2"] == 0) & (df["AskVolume_2"] == 0) & \
          (df["AskPrice_3"]  == 0) & (df["AskVolume_3"] == 0) & (df["AskPrice_4"] == 0) & \
          (df["AskVolume_4"] == 0) & (df["AskPrice_5"] == 0) & (df["AskVolume_5"] > 0)

    a = df.loc[con].index.to_numpy()
    df.loc[a,["Type"]] = "Limit"
    df.loc[a,["Quote"]] = 5
    df.loc[a,["Sign"]] = -1
    df.loc[a,["Price"]] = clean_data.loc[a]["AskPrice_5"]
    df.loc[a,["Volume"]] = df.loc[a]["AskVolume_5"]

    # AskVolume_5 < 0
    con = (df["AskPrice_0"] == 0) & (df["AskVolume_0"] == 0) & (df["AskPrice_1"] == 0) & \
          (df["AskVolume_1"] == 0) & (df["AskPrice_2"] == 0) & (df["AskVolume_2"] == 0) & \
          (df["AskPrice_3"]  == 0) & (df["AskVolume_3"] == 0) & (df["AskPrice_4"] == 0) & \
          (df["AskVolume_4"] == 0) & (df["AskPrice_5"] == 0) & (df["AskVolume_5"] < 0)

    a = df.loc[con].index.to_numpy()
    df.loc[a,["Type"]] = "Market/Cancel"
    df.loc[a,["Quote"]] = 5
    df.loc[a,["Sign"]] = -1
    df.loc[a,["Price"]] = clean_data.loc[a]["AskPrice_5"]
    df.loc[a,["Volume"]] = df.loc[a]["AskVolume_5"]* (-1)

    # BidPrice_6 > 0
    con = (df["BidPrice_0"] == 0) & (df["BidVolume_0"] == 0) & (df["BidPrice_1"] == 0) & \
          (df["BidVolume_1"] == 0) & (df["BidPrice_2"] == 0) & (df["BidVolume_2"] == 0) & \
          (df["BidPrice_3"] == 0) & (df["BidVolume_3"] == 0) & (df["BidPrice_4"] == 0) & \
          (df["BidVolume_4"] == 0) & (df["BidPrice_5"] == 0) & (df["BidVolume_5"] == 0) & \
          (df["BidPrice_6"] > 0)

    a = df.loc[con].index.to_numpy()
    df.loc[a,["Type"]] = "Limit"
    df.loc[a,["Quote"]] = 6
    df.loc[a,["Sign"]] = 1
    df.loc[a,["Price"]] = clean_data.loc[a]["BidPrice_6"]
    df.loc[a,["Volume"]] = clean_data.loc[a]["BidVolume_6"]

    # BidPrice_6 < 0
    con = (df["BidPrice_0"] == 0) & (df["BidVolume_0"] == 0) & (df["BidPrice_1"] == 0) & \
          (df["BidVolume_1"] == 0) & (df["BidPrice_2"] == 0) & (df["BidVolume_2"] == 0) & \
          (df["BidPrice_3"] == 0) & (df["BidVolume_3"] == 0) & (df["BidPrice_4"] == 0) & \
          (df["BidVolume_4"] == 0) & (df["BidPrice_5"] == 0) & (df["BidVolume_5"] == 0) & \
          (df["BidPrice_6"] < 0)

    a = df.loc[con].index.to_numpy()
    df.loc[a,["Type"]] = "Market/Cancel"
    df.loc[a,["Quote"]] = 6
    df.loc[a,["Sign"]] = 1
    df.loc[a,["Price"]] = clean_data.loc[a-1]["BidPrice_6"].to_list()
    df.loc[a,["Volume"]] = clean_data.loc[a-1]["BidVolume_6"].to_list()

    # AskPrice_6 > 0
    con = (df["AskPrice_0"] == 0) & (df["AskVolume_0"] == 0) & (df["AskPrice_1"] == 0) & \
          (df["AskVolume_1"] == 0) & (df["AskPrice_2"] == 0) & (df["AskVolume_2"] == 0) & \
          (df["AskPrice_3"]  == 0) & (df["AskVolume_3"] == 0) & (df["AskPrice_4"] == 0) & \
          (df["AskVolume_4"] == 0) & (df["AskPrice_5"] == 0) & (df["AskVolume_5"] == 0) & \
          (df["AskPrice_6"] > 0)

    a = df.loc[con].index.to_numpy()
    df.loc[a,["Type"]] = "Market/Cancel"
    df.loc[a,["Quote"]] = 6
    df.loc[a,["Sign"]] = -1
    df.loc[a,["Price"]] = clean_data.loc[a-1]["AskPrice_6"].to_list()
    df.loc[a,["Volume"]] = clean_data.loc[a-1]["AskVolume_6"].to_list()

    # AskPrice_6 < 0
    con = (df["AskPrice_0"] == 0) & (df["AskVolume_0"] == 0) & (df["AskPrice_1"] == 0) & \
          (df["AskVolume_1"] == 0) & (df["AskPrice_2"] == 0) & (df["AskVolume_2"] == 0) & \
          (df["AskPrice_3"]  == 0) & (df["AskVolume_3"] == 0) & (df["AskPrice_4"] == 0) & \
          (df["AskVolume_4"] == 0) & (df["AskPrice_5"] == 0) & (df["AskVolume_5"] == 0) & \
          (df["AskPrice_6"] < 0)

    a = df.loc[con].index.to_numpy()
    df.loc[a,["Type"]] = "Limit"
    df.loc[a,["Quote"]] = 6
    df.loc[a,["Sign"]] = -1
    df.loc[a,["Price"]] = clean_data.loc[a]["AskPrice_6"]
    df.loc[a,["Volume"]] = clean_data.loc[a]["AskVolume_6"]


    # BidVolume_6 > 0
    con = (df["BidPrice_0"] == 0) & (df["BidVolume_0"] == 0) & (df["BidPrice_1"] == 0) & \
          (df["BidVolume_1"] == 0) & (df["BidPrice_2"] == 0) & (df["BidVolume_2"] == 0) & \
          (df["BidPrice_3"] == 0) & (df["BidVolume_3"] == 0) & (df["BidPrice_4"] == 0) & \
          (df["BidVolume_4"] == 0) & (df["BidPrice_5"] == 0) & (df["BidVolume_5"] == 0) & \
          (df["BidPrice_6"] == 0) & (df["BidVolume_6"] > 0)

    a = df.loc[con].index.to_numpy()
    df.loc[a,["Type"]] = "Limit"
    df.loc[a,["Quote"]] = 6
    df.loc[a,["Sign"]] = 1
    df.loc[a,["Price"]] = clean_data.loc[a]["BidPrice_6"]
    df.loc[a,["Volume"]] = df.loc[a]["BidVolume_6"]

    # BidVolume_6 < 0
    con = (df["BidPrice_0"] == 0) & (df["BidVolume_0"] == 0) & (df["BidPrice_1"] == 0) & \
          (df["BidVolume_1"] == 0)  & (df["BidPrice_2"] == 0) & (df["BidVolume_2"] == 0) & \
          (df["BidPrice_3"] == 0) & (df["BidVolume_3"] == 0) & (df["BidPrice_4"] == 0) & \
          (df["BidVolume_4"] == 0) & (df["BidPrice_5"] == 0) & (df["BidVolume_5"] == 0) & \
          (df["BidPrice_6"] == 0) & (df["BidVolume_6"] < 0)

    a = df.loc[con].index.to_numpy()
    df.loc[a,["Type"]] = "Market/Cancel"
    df.loc[a,["Quote"]] = 6
    df.loc[a,["Sign"]] = 1
    df.loc[a,["Price"]] = clean_data.loc[a]["BidPrice_6"]
    df.loc[a,["Volume"]] = df.loc[a]["BidVolume_6"] * (-1)

    # AskVolume_6 > 0
    con = (df["AskPrice_0"] == 0) & (df["AskVolume_0"] == 0) & (df["AskPrice_1"] == 0) & \
          (df["AskVolume_1"] == 0) & (df["AskPrice_2"] == 0) & (df["AskVolume_2"] == 0) & \
          (df["AskPrice_3"]  == 0) & (df["AskVolume_3"] == 0) & (df["AskPrice_4"] == 0) & \
          (df["AskVolume_4"] == 0) & (df["AskPrice_5"] == 0) & (df["AskVolume_5"] == 0) & \
          (df["AskPrice_6"] == 0) & (df["AskVolume_6"] > 0)

    a = df.loc[con].index.to_numpy()
    df.loc[a,["Type"]] = "Limit"
    df.loc[a,["Quote"]] = 6
    df.loc[a,["Sign"]] = -1
    df.loc[a,["Price"]] = clean_data.loc[a]["AskPrice_6"]
    df.loc[a,["Volume"]] = df.loc[a]["AskVolume_6"]

    # AskVolume_6 < 0
    con = (df["AskPrice_0"] == 0) & (df["AskVolume_0"] == 0) & (df["AskPrice_1"] == 0) & \
          (df["AskVolume_1"] == 0) & (df["AskPrice_2"] == 0) & (df["AskVolume_2"] == 0) & \
          (df["AskPrice_3"]  == 0) & (df["AskVolume_3"] == 0) & (df["AskPrice_4"] == 0) & \
          (df["AskVolume_4"] == 0) & (df["AskPrice_5"] == 0) & (df["AskVolume_5"] == 0) & \
          (df["AskPrice_6"] == 0) & (df["AskVolume_6"] < 0)

    a = df.loc[con].index.to_numpy()
    df.loc[a,["Type"]] = "Market/Cancel"
    df.loc[a,["Quote"]] = 6
    df.loc[a,["Sign"]] = -1
    df.loc[a,["Price"]] = clean_data.loc[a]["AskPrice_6"]
    df.loc[a,["Volume"]] = df.loc[a]["AskVolume_6"]* (-1)

    # BidPrice_7 > 0
    con = (df["BidPrice_0"] == 0) & (df["BidVolume_0"] == 0) & (df["BidPrice_1"] == 0) & \
          (df["BidVolume_1"] == 0) & (df["BidPrice_2"] == 0) & (df["BidVolume_2"] == 0) & \
          (df["BidPrice_3"] == 0) & (df["BidVolume_3"] == 0) & (df["BidPrice_4"] == 0) & \
          (df["BidVolume_4"] == 0) & (df["BidPrice_5"] == 0) & (df["BidVolume_5"] == 0) & \
          (df["BidPrice_6"] == 0) & (df["BidVolume_6"] == 0) & (df["BidPrice_7"] > 0)

    a = df.loc[con].index.to_numpy()
    df.loc[a,["Type"]] = "Limit"
    df.loc[a,["Quote"]] = 7
    df.loc[a,["Sign"]] = 1
    df.loc[a,["Price"]] = clean_data.loc[a]["BidPrice_7"]
    df.loc[a,["Volume"]] = clean_data.loc[a]["BidVolume_7"]

    # BidPrice_7 < 0
    con = (df["BidPrice_0"] == 0) & (df["BidVolume_0"] == 0) & (df["BidPrice_1"] == 0) & \
          (df["BidVolume_1"] == 0) & (df["BidPrice_2"] == 0) & (df["BidVolume_2"] == 0) & \
          (df["BidPrice_3"] == 0) & (df["BidVolume_3"] == 0) & (df["BidPrice_4"] == 0) & \
          (df["BidVolume_4"] == 0) & (df["BidPrice_5"] == 0) & (df["BidVolume_5"] == 0) & \
          (df["BidPrice_6"] == 0) & (df["BidVolume_6"] == 0) & (df["BidPrice_7"] < 0)

    a = df.loc[con].index.to_numpy()
    df.loc[a,["Type"]] = "Market/Cancel"
    df.loc[a,["Quote"]] = 7
    df.loc[a,["Sign"]] = 1
    df.loc[a,["Price"]] = clean_data.loc[a-1]["BidPrice_7"].to_list()
    df.loc[a,["Volume"]] = clean_data.loc[a-1]["BidVolume_7"].to_list()

    # AskPrice_7 > 0
    con = (df["AskPrice_0"] == 0) & (df["AskVolume_0"] == 0) & (df["AskPrice_1"] == 0) & \
          (df["AskVolume_1"] == 0) & (df["AskPrice_2"] == 0) & (df["AskVolume_2"] == 0) & \
          (df["AskPrice_3"]  == 0) & (df["AskVolume_3"] == 0) & (df["AskPrice_4"] == 0) & \
          (df["AskVolume_4"] == 0) & (df["AskPrice_5"] == 0) & (df["AskVolume_5"] == 0) & \
          (df["AskPrice_6"] == 0) & (df["AskVolume_6"] == 0) & (df["AskPrice_7"] > 0)

    a = df.loc[con].index.to_numpy()
    df.loc[a,["Type"]] = "Market/Cancel"
    df.loc[a,["Quote"]] = 7
    df.loc[a,["Sign"]] = -1
    df.loc[a,["Price"]] = clean_data.loc[a-1]["AskPrice_7"].to_list()
    df.loc[a,["Volume"]] = clean_data.loc[a-1]["AskVolume_7"].to_list()

    # AskPrice_7 < 0
    con = (df["AskPrice_0"] == 0) & (df["AskVolume_0"] == 0) & (df["AskPrice_1"] == 0) & \
          (df["AskVolume_1"] == 0) & (df["AskPrice_2"] == 0) & (df["AskVolume_2"] == 0) & \
          (df["AskPrice_3"]  == 0) & (df["AskVolume_3"] == 0) & (df["AskPrice_4"] == 0) & \
          (df["AskVolume_4"] == 0) & (df["AskPrice_5"] == 0) & (df["AskVolume_5"] == 0) & \
          (df["AskPrice_6"] == 0) & (df["AskVolume_6"] == 0) & (df["AskPrice_7"] < 0)

    a = df.loc[con].index.to_numpy()
    df.loc[a,["Type"]] = "Limit"
    df.loc[a,["Quote"]] = 7
    df.loc[a,["Sign"]] = -1
    df.loc[a,["Price"]] = clean_data.loc[a]["AskPrice_7"]
    df.loc[a,["Volume"]] = clean_data.loc[a]["AskVolume_7"]


    # BidVolume_7 > 0
    con = (df["BidPrice_0"] == 0) & (df["BidVolume_0"] == 0) & (df["BidPrice_1"] == 0) & \
          (df["BidVolume_1"] == 0) & (df["BidPrice_2"] == 0) & (df["BidVolume_2"] == 0) & \
          (df["BidPrice_3"] == 0) & (df["BidVolume_3"] == 0) & (df["BidPrice_4"] == 0) & \
          (df["BidVolume_4"] == 0) & (df["BidPrice_5"] == 0) & (df["BidVolume_5"] == 0) & \
          (df["BidPrice_6"] == 0) & (df["BidVolume_6"] == 0) & (df["BidPrice_7"] == 0) & \
          (df["BidVolume_7"] > 0)

    a = df.loc[con].index.to_numpy()
    df.loc[a,["Type"]] = "Limit"
    df.loc[a,["Quote"]] = 7
    df.loc[a,["Sign"]] = 1
    df.loc[a,["Price"]] = clean_data.loc[a]["BidPrice_7"]
    df.loc[a,["Volume"]] = df.loc[a]["BidVolume_7"]

    # BidVolume_7 < 0
    con = (df["BidPrice_0"] == 0) & (df["BidVolume_0"] == 0) & (df["BidPrice_1"] == 0) & \
          (df["BidVolume_1"] == 0) & (df["BidPrice_2"] == 0) & (df["BidVolume_2"] == 0) & \
          (df["BidPrice_3"] == 0) & (df["BidVolume_3"] == 0) & (df["BidPrice_4"] == 0) & \
          (df["BidVolume_4"] == 0) & (df["BidPrice_5"] == 0) & (df["BidVolume_5"] == 0) & \
          (df["BidPrice_6"] == 0) & (df["BidVolume_6"] == 0) & (df["BidPrice_7"] == 0) & \
          (df["BidVolume_7"] < 0)

    a = df.loc[con].index.to_numpy()
    df.loc[a,["Type"]] = "Market/Cancel"
    df.loc[a,["Quote"]] = 7
    df.loc[a,["Sign"]] = 1
    df.loc[a,["Price"]] = clean_data.loc[a]["BidPrice_7"]
    df.loc[a,["Volume"]] = df.loc[a]["BidVolume_7"] * (-1)

    # AskVolume_7 > 0
    con = (df["AskPrice_0"] == 0) & (df["AskVolume_0"] == 0) & (df["AskPrice_1"] == 0) & \
          (df["AskVolume_1"] == 0) & (df["AskPrice_2"] == 0) & (df["AskVolume_2"] == 0) & \
          (df["AskPrice_3"]  == 0) & (df["AskVolume_3"] == 0) & (df["AskPrice_4"] == 0) & \
          (df["AskVolume_4"] == 0) & (df["AskPrice_5"] == 0) & (df["AskVolume_5"] == 0) & \
          (df["AskPrice_6"] == 0) & (df["AskVolume_6"] == 0) & (df["AskPrice_7"] == 0) & \
          (df["AskVolume_7"] > 0)

    a = df.loc[con].index.to_numpy()
    df.loc[a,["Type"]] = "Limit"
    df.loc[a,["Quote"]] = 7
    df.loc[a,["Sign"]] = -1
    df.loc[a,["Price"]] = clean_data.loc[a]["AskPrice_7"]
    df.loc[a,["Volume"]] = df.loc[a]["AskVolume_7"]

    # AskVolume_7 < 0
    con = (df["AskPrice_0"] == 0) & (df["AskVolume_0"] == 0) & (df["AskPrice_1"] == 0) & \
          (df["AskVolume_1"] == 0)  & (df["AskPrice_2"] == 0) & (df["AskVolume_2"] == 0) & \
          (df["AskPrice_3"]  == 0) & (df["AskVolume_3"] == 0) & (df["AskPrice_4"] == 0) & \
          (df["AskVolume_4"] == 0) & (df["AskPrice_5"] == 0) & (df["AskVolume_5"] == 0) & \
          (df["AskPrice_6"] == 0) & (df["AskVolume_6"] == 0) & (df["AskPrice_7"] == 0) & \
          (df["AskVolume_7"] < 0)

    a = df.loc[con].index.to_numpy()
    df.loc[a,["Type"]] = "Market/Cancel"
    df.loc[a,["Quote"]] = 7
    df.loc[a,["Sign"]] = -1
    df.loc[a,["Price"]] = clean_data.loc[a]["AskPrice_7"]
    df.loc[a,["Volume"]] = df.loc[a]["AskVolume_7"]* (-1)

    # BidPrice_8 > 0
    con = (df["BidPrice_0"] == 0) & (df["BidVolume_0"] == 0) & (df["BidPrice_1"] == 0) & \
          (df["BidVolume_1"] == 0) & (df["BidPrice_2"] == 0) & (df["BidVolume_2"] == 0) & \
          (df["BidPrice_3"] == 0) & (df["BidVolume_3"] == 0) & (df["BidPrice_4"] == 0) & \
          (df["BidVolume_4"] == 0) & (df["BidPrice_5"] == 0) & (df["BidVolume_5"] == 0) & \
          (df["BidPrice_6"] == 0) & (df["BidVolume_6"] == 0) & (df["BidPrice_7"] == 0) & \
          (df["BidVolume_7"] == 0) & (df["BidPrice_8"] > 0)

    a = df.loc[con].index.to_numpy()
    df.loc[a,["Type"]] = "Limit"
    df.loc[a,["Quote"]] = 8
    df.loc[a,["Sign"]] = 1
    df.loc[a,["Price"]] = clean_data.loc[a]["BidPrice_8"]
    df.loc[a,["Volume"]] = clean_data.loc[a]["BidVolume_8"]

    # BidPrice_8 < 0
    con = (df["BidPrice_0"] == 0) & (df["BidVolume_0"] == 0) & (df["BidPrice_1"] == 0) & \
          (df["BidVolume_1"] == 0) & (df["BidPrice_2"] == 0) & (df["BidVolume_2"] == 0) & \
          (df["BidPrice_3"] == 0) & (df["BidVolume_3"] == 0) & (df["BidPrice_4"] == 0) & \
          (df["BidVolume_4"] == 0) & (df["BidPrice_5"] == 0) & (df["BidVolume_5"] == 0) & \
          (df["BidPrice_6"] == 0) & (df["BidVolume_6"] == 0) & (df["BidPrice_7"] == 0) & \
          (df["BidVolume_7"] == 0) & (df["BidPrice_8"] < 0)

    a = df.loc[con].index.to_numpy()
    df.loc[a,["Type"]] = "Market/Cancel"
    df.loc[a,["Quote"]] = 8
    df.loc[a,["Sign"]] = 1
    df.loc[a,["Price"]] = clean_data.loc[a-1]["BidPrice_8"].to_list()
    df.loc[a,["Volume"]] = clean_data.loc[a-1]["BidVolume_8"].to_list()

    # AskPrice_8 > 0
    con = (df["AskPrice_0"] == 0) & (df["AskVolume_0"] == 0) & (df["AskPrice_1"] == 0) & \
          (df["AskVolume_1"] == 0) & (df["AskPrice_2"] == 0) & (df["AskVolume_2"] == 0) & \
          (df["AskPrice_3"] == 0) & (df["AskVolume_3"] == 0) & (df["AskPrice_4"] == 0) & \
          (df["AskVolume_4"] == 0) & (df["AskPrice_5"] == 0) & (df["AskVolume_5"] == 0) & \
          (df["AskPrice_6"] == 0) & (df["AskVolume_6"] == 0) & (df["AskPrice_7"] == 0) & \
          (df["AskVolume_7"] == 0) & (df["AskPrice_8"] > 0)

    a = df.loc[con].index.to_numpy()
    df.loc[a,["Type"]] = "Market/Cancel"
    df.loc[a,["Quote"]] = 8
    df.loc[a,["Sign"]] = -1
    df.loc[a,["Price"]] = clean_data.loc[a-1]["AskPrice_8"].to_list()
    df.loc[a,["Volume"]] = clean_data.loc[a-1]["AskVolume_8"].to_list()

    # AskPrice_8 < 0
    con = (df["AskPrice_0"] == 0) & (df["AskVolume_0"] == 0) & (df["AskPrice_1"] == 0) & \
          (df["AskVolume_1"] == 0) & (df["AskPrice_2"] == 0) & (df["AskVolume_2"] == 0) & \
          (df["AskPrice_3"] == 0) & (df["AskVolume_3"] == 0) & (df["AskPrice_4"] == 0) & \
          (df["AskVolume_4"] == 0) & (df["AskPrice_5"] == 0) & (df["AskVolume_5"] == 0) & \
          (df["AskPrice_6"] == 0) & (df["AskVolume_6"] == 0) & (df["AskPrice_7"] == 0) & \
          (df["AskVolume_7"] == 0) & (df["AskPrice_8"] < 0)

    a = df.loc[con].index.to_numpy()
    df.loc[a,["Type"]] = "Limit"
    df.loc[a,["Quote"]] = 8
    df.loc[a,["Sign"]] = -1
    df.loc[a,["Price"]] = clean_data.loc[a]["AskPrice_8"]
    df.loc[a,["Volume"]] = clean_data.loc[a]["AskVolume_8"]


    # BidVolume_8 > 0
    con = (df["BidPrice_0"] == 0) & (df["BidVolume_0"] == 0) & (df["BidPrice_1"] == 0) & \
          (df["BidVolume_1"] == 0) & (df["BidPrice_2"] == 0) & (df["BidVolume_2"] == 0) & \
          (df["BidPrice_3"] == 0) & (df["BidVolume_3"] == 0) & (df["BidPrice_4"] == 0) & \
          (df["BidVolume_4"] == 0) & (df["BidPrice_5"] == 0) & (df["BidVolume_5"] == 0) & \
          (df["BidPrice_6"] == 0) & (df["BidVolume_6"] == 0) & (df["BidPrice_7"] == 0) & \
          (df["BidVolume_7"] == 0) & (df["BidPrice_8"] == 0) & (df["BidVolume_8"] > 0)

    a = df.loc[con].index.to_numpy()
    df.loc[a,["Type"]] = "Limit"
    df.loc[a,["Quote"]] = 8
    df.loc[a,["Sign"]] = 1
    df.loc[a,["Price"]] = clean_data.loc[a]["BidPrice_8"]
    df.loc[a,["Volume"]] = df.loc[a]["BidVolume_8"]

    # BidVolume_8 < 0
    con = (df["BidPrice_0"] == 0) & (df["BidVolume_0"] == 0) & (df["BidPrice_1"] == 0) & \
          (df["BidVolume_1"] == 0) & (df["BidPrice_2"] == 0) & (df["BidVolume_2"] == 0) & \
          (df["BidPrice_3"] == 0) & (df["BidVolume_3"] == 0) & (df["BidPrice_4"] == 0) & \
          (df["BidVolume_4"] == 0) & (df["BidPrice_5"] == 0) & (df["BidVolume_5"] == 0) & \
          (df["BidPrice_6"] == 0) & (df["BidVolume_6"] == 0) & (df["BidPrice_7"] == 0) & \
          (df["BidVolume_7"] == 0) & (df["BidPrice_8"] == 0) & (df["BidVolume_8"] < 0)

    a = df.loc[con].index.to_numpy()
    df.loc[a,["Type"]] = "Market/Cancel"
    df.loc[a,["Quote"]] = 8
    df.loc[a,["Sign"]] = 1
    df.loc[a,["Price"]] = clean_data.loc[a]["BidPrice_8"]
    df.loc[a,["Volume"]] = df.loc[a]["BidVolume_8"] * (-1)

    # AskVolume_8 > 0
    con = (df["AskPrice_0"] == 0) & (df["AskVolume_0"] == 0) & (df["AskPrice_1"] == 0) & \
          (df["AskVolume_1"] == 0) & (df["AskPrice_2"] == 0) & (df["AskVolume_2"] == 0) & \
          (df["AskPrice_3"] == 0) & (df["AskVolume_3"] == 0) & (df["AskPrice_4"] == 0) & \
          (df["AskVolume_4"] == 0) & (df["AskPrice_5"] == 0) & (df["AskVolume_5"] == 0) & \
          (df["AskPrice_6"] == 0) & (df["AskVolume_6"] == 0) & (df["AskPrice_7"] == 0) & \
          (df["AskVolume_7"] == 0) & (df["AskPrice_8"] == 0) & (df["AskVolume_8"] > 0)

    a = df.loc[con].index.to_numpy()
    df.loc[a,["Type"]] = "Limit"
    df.loc[a,["Quote"]] = 8
    df.loc[a,["Sign"]] = -1
    df.loc[a,["Price"]] = clean_data.loc[a]["AskPrice_8"]
    df.loc[a,["Volume"]] = df.loc[a]["AskVolume_8"]

    # AskVolume_8 < 0
    con = (df["AskPrice_0"] == 0) & (df["AskVolume_0"] == 0) & (df["AskPrice_1"] == 0) & \
          (df["AskVolume_1"] == 0) & (df["AskPrice_2"] == 0) & (df["AskVolume_2"] == 0) & \
          (df["AskPrice_3"] == 0) & (df["AskVolume_3"] == 0) & (df["AskPrice_4"] == 0) & \
          (df["AskVolume_4"] == 0) & (df["AskPrice_5"] == 0) & (df["AskVolume_5"] == 0) & \
          (df["AskPrice_6"] == 0) & (df["AskVolume_6"] == 0) & (df["AskPrice_7"] == 0) & \
          (df["AskVolume_7"] == 0) & (df["AskPrice_8"] == 0) & (df["AskVolume_8"] < 0)

    a = df.loc[con].index.to_numpy()
    df.loc[a,["Type"]] = "Market/Cancel"
    df.loc[a,["Quote"]] = 8
    df.loc[a,["Sign"]] = -1
    df.loc[a,["Price"]] = clean_data.loc[a]["AskPrice_8"]
    df.loc[a,["Volume"]] = df.loc[a]["AskVolume_8"]* (-1)

    # BidPrice_9 > 0
    con = (df["BidPrice_0"] == 0) & (df["BidVolume_0"] == 0) & (df["BidPrice_1"] == 0) & \
          (df["BidVolume_1"] == 0) & (df["BidPrice_2"] == 0) & (df["BidVolume_2"] == 0) & \
          (df["BidPrice_3"] == 0) & (df["BidVolume_3"] == 0) & (df["BidPrice_4"] == 0) & \
          (df["BidVolume_4"] == 0) & (df["BidPrice_5"] == 0) & (df["BidVolume_5"] == 0) & \
          (df["BidPrice_6"] == 0) & (df["BidVolume_6"] == 0) & (df["BidPrice_7"] == 0) & \
          (df["BidVolume_7"] == 0) & (df["BidPrice_8"] == 0) & (df["BidVolume_8"] == 0) & \
          (df["BidPrice_9"] > 0)

    a = df.loc[con].index.to_numpy()
    df.loc[a,["Type"]] = "Limit"
    df.loc[a,["Quote"]] = 9
    df.loc[a,["Sign"]] = 1
    df.loc[a,["Price"]] = clean_data.loc[a]["BidPrice_9"]
    df.loc[a,["Volume"]] = clean_data.loc[a]["BidVolume_9"]

    # BidPrice_9 < 0
    con = (df["BidPrice_0"] == 0) & (df["BidVolume_0"] == 0) & (df["BidPrice_1"] == 0) & \
          (df["BidVolume_1"] == 0) & (df["BidPrice_2"] == 0) & (df["BidVolume_2"] == 0) & \
          (df["BidPrice_3"] == 0) & (df["BidVolume_3"] == 0) & (df["BidPrice_4"] == 0) & \
          (df["BidVolume_4"] == 0) & (df["BidPrice_5"] == 0) & (df["BidVolume_5"] == 0) & \
          (df["BidPrice_6"] == 0) & (df["BidVolume_6"] == 0) & (df["BidPrice_7"] == 0) & \
          (df["BidVolume_7"] == 0) & (df["BidPrice_8"] == 0) & (df["BidVolume_8"] == 0) & \
          (df["BidPrice_9"] < 0)

    a = df.loc[con].index.to_numpy()
    df.loc[a,["Type"]] = "Market/Cancel"
    df.loc[a,["Quote"]] = 9
    df.loc[a,["Sign"]] = 1
    df.loc[a,["Price"]] = clean_data.loc[a-1]["BidPrice_9"].to_list()
    df.loc[a,["Volume"]] = clean_data.loc[a-1]["BidVolume_9"].to_list()

    # AskPrice_9 > 0
    con = (df["AskPrice_0"] == 0) & (df["AskVolume_0"] == 0) & (df["AskPrice_1"] == 0) & \
          (df["AskVolume_1"] == 0) & (df["AskPrice_2"] == 0) & (df["AskVolume_2"] == 0) & \
          (df["AskPrice_3"] == 0) & (df["AskVolume_3"] == 0) & (df["AskPrice_4"] == 0) & \
          (df["AskVolume_4"] == 0) & (df["AskPrice_5"] == 0) & (df["AskVolume_5"] == 0) & \
          (df["AskPrice_6"] == 0) & (df["AskVolume_6"] == 0) & (df["AskPrice_7"] == 0) & \
          (df["AskVolume_7"] == 0) & (df["AskPrice_8"] == 0) & (df["AskVolume_8"] == 0) & \
          (df["AskPrice_9"] > 0)

    a = df.loc[con].index.to_numpy()
    df.loc[a,["Type"]] = "Market/Cancel"
    df.loc[a,["Quote"]] = 9
    df.loc[a,["Sign"]] = -1
    df.loc[a,["Price"]] = clean_data.loc[a-1]["AskPrice_9"].to_list()
    df.loc[a,["Volume"]] = clean_data.loc[a-1]["AskVolume_9"].to_list()

    # AskPrice_9 < 0
    con = (df["AskPrice_0"] == 0) & (df["AskVolume_0"] == 0) & (df["AskPrice_1"] == 0) & \
          (df["AskVolume_1"] == 0) & (df["AskPrice_2"] == 0) & (df["AskVolume_2"] == 0) & \
          (df["AskPrice_3"] == 0) & (df["AskVolume_3"] == 0) & (df["AskPrice_4"] == 0) & \
          (df["AskVolume_4"] == 0) & (df["AskPrice_5"] == 0) & (df["AskVolume_5"] == 0) & \
          (df["AskPrice_6"] == 0) & (df["AskVolume_6"] == 0) & (df["AskPrice_7"] == 0) & \
          (df["AskVolume_7"] == 0) & (df["AskPrice_8"] == 0) & (df["AskVolume_8"] == 0) & \
          (df["AskPrice_9"] < 0)

    a = df.loc[con].index.to_numpy()
    df.loc[a,["Type"]] = "Limit"
    df.loc[a,["Quote"]] = 9
    df.loc[a,["Sign"]] = -1
    df.loc[a,["Price"]] = clean_data.loc[a]["AskPrice_9"]
    df.loc[a,["Volume"]] = clean_data.loc[a]["AskVolume_9"]


    # BidVolume_9 > 0
    con = (df["BidPrice_0"] == 0) & (df["BidVolume_0"] == 0) & (df["BidPrice_1"] == 0) & \
          (df["BidVolume_1"] == 0) & (df["BidPrice_2"] == 0) & (df["BidVolume_2"] == 0) & \
          (df["BidPrice_3"] == 0) & (df["BidVolume_3"] == 0) & (df["BidPrice_4"] == 0) & \
          (df["BidVolume_4"] == 0) & (df["BidPrice_5"] == 0) & (df["BidVolume_5"] == 0) & \
          (df["BidPrice_6"] == 0) & (df["BidVolume_6"] == 0) & (df["BidPrice_7"] == 0) & \
          (df["BidVolume_7"] == 0) & (df["BidPrice_8"] == 0) & (df["BidVolume_8"] == 0) & \
          (df["BidPrice_9"] == 0) & (df["BidVolume_9"] > 0)

    a = df.loc[con].index.to_numpy()
    df.loc[a,["Type"]] = "Limit"
    df.loc[a,["Quote"]] = 9
    df.loc[a,["Sign"]] = 1
    df.loc[a,["Price"]] = clean_data.loc[a]["BidPrice_9"]
    df.loc[a,["Volume"]] = df.loc[a]["BidVolume_9"]

    # BidVolume_9 < 0
    con = (df["BidPrice_0"] == 0) & (df["BidVolume_0"] == 0) & (df["BidPrice_1"] == 0) & \
          (df["BidVolume_1"] == 0) & (df["BidPrice_2"] == 0) & (df["BidVolume_2"] == 0) & \
          (df["BidPrice_3"] == 0) & (df["BidVolume_3"] == 0) & (df["BidPrice_4"] == 0) & \
          (df["BidVolume_4"] == 0) & (df["BidPrice_5"] == 0) & (df["BidVolume_5"] == 0) & \
          (df["BidPrice_6"] == 0) & (df["BidVolume_6"] == 0) & (df["BidPrice_7"] == 0) & \
          (df["BidVolume_7"] == 0) & (df["BidPrice_8"] == 0) & (df["BidVolume_8"] == 0) & \
          (df["BidPrice_9"] == 0) & (df["BidVolume_9"] < 0)

    a = df.loc[con].index.to_numpy()
    df.loc[a,["Type"]] = "Market/Cancel"
    df.loc[a,["Quote"]] = 9
    df.loc[a,["Sign"]] = 1
    df.loc[a,["Price"]] = clean_data.loc[a]["BidPrice_9"]
    df.loc[a,["Volume"]] = df.loc[a]["BidVolume_9"] * (-1)

    # AskVolume_9 > 0
    con = (df["AskPrice_0"] == 0) & (df["AskVolume_0"] == 0) & (df["AskPrice_1"] == 0) & \
          (df["AskVolume_1"] == 0) & (df["AskPrice_2"] == 0) & (df["AskVolume_2"] == 0) & \
          (df["AskPrice_3"] == 0) & (df["AskVolume_3"] == 0) & (df["AskPrice_4"] == 0) & \
          (df["AskVolume_4"] == 0) & (df["AskPrice_5"] == 0) & (df["AskVolume_5"] == 0) & \
          (df["AskPrice_6"] == 0) & (df["AskVolume_6"] == 0) & (df["AskPrice_7"] == 0) & \
          (df["AskVolume_7"] == 0) & (df["AskPrice_8"] == 0) & (df["AskVolume_8"] == 0) & \
          (df["AskPrice_9"] == 0) & (df["AskVolume_9"] > 0)

    a = df.loc[con].index.to_numpy()
    df.loc[a,["Type"]] = "Limit"
    df.loc[a,["Quote"]] = 9
    df.loc[a,["Sign"]] = -1
    df.loc[a,["Price"]] = clean_data.loc[a]["AskPrice_9"]
    df.loc[a,["Volume"]] = df.loc[a]["AskVolume_9"]

    # AskVolume_9 < 0
    con = (df["AskPrice_0"] == 0) & (df["AskVolume_0"] == 0) & (df["AskPrice_1"] == 0) & \
          (df["AskVolume_1"] == 0) & (df["AskPrice_2"] == 0) & (df["AskVolume_2"] == 0) & \
          (df["AskPrice_3"] == 0) & (df["AskVolume_3"] == 0) & (df["AskPrice_4"] == 0) & \
          (df["AskVolume_4"] == 0) & (df["AskPrice_5"] == 0) & (df["AskVolume_5"] == 0) & \
          (df["AskPrice_6"] == 0) & (df["AskVolume_6"] == 0) & (df["AskPrice_7"] == 0) & \
          (df["AskVolume_7"] == 0) & (df["AskPrice_8"] == 0) & (df["AskVolume_8"] == 0) & \
          (df["AskPrice_9"] == 0) & (df["AskVolume_9"] < 0)

    a = df.loc[con].index.to_numpy()
    df.loc[a,["Type"]] = "Market/Cancel"
    df.loc[a,["Quote"]] = 9
    df.loc[a,["Sign"]] = -1
    df.loc[a,["Price"]] = clean_data.loc[a]["AskPrice_9"]
    df.loc[a,["Volume"]] = df.loc[a]["AskVolume_9"]* (-1)

    # questa parte serve perché se c'è una cancellazione/trade che rimuove completamente
    # tutti gli ordini da una quota dell'ask arrivo a risultati sbagliati
    a = df[(df["Price"]==0) & (df["Quote"] != "NoBest")].index.to_list()

    for i in a:
        if df["Type"].at[i] == "Market/Cancel":
            q = df["Quote"].at[i]
            df["Type"].at[i] = "Limit"
            df["Price"].at[i] = clean_data["AskPrice_" + str(q)].at[i]
            df["Volume"].at[i] = clean_data["AskPrice_" + str(q)].at[i]

        elif df["Type"].at[i] == "Limit":
            q = df["Quote"].at[i]
            df["Type"].at[i] = "Market/Cancel"
            df["Price"].at[i] = clean_data["AskPrice_" + str(q)].at[i-1]
            df["Volume"].at[i] = clean_data["AskPrice_" + str(q)].at[i-1]


    new_df = df.loc[:,["Price", "Volume", "Sign", "Quote", "Type"]]
    new_df["DateTime"] = clean_data["Datetime"]
    new_df["Seconds"] = clean_data["Seconds"]
    new_df["Spread"] = clean_data["Spread"]
    new_df["MidPrice"] = clean_data["MidPrice"]
    new_df["AskVolume_0"] = clean_data["AskVolume_0"]
    new_df["BidVolume_0"] = clean_data["BidVolume_0"]
    new_df["Type"].replace([0], "NoBest", inplace = True)

    new_df.drop(0, inplace = True)
    new_df.reset_index(drop = True, inplace = True)

    return new_df

def load_trade_data(filepath, start_month = False):

    # import data
    df = pd.read_csv(filepath)
    df.drop(columns = ["Unnamed: 0"], inplace = True)
    # transform the column Datetime from string to datetime
    df["DateTime"]= pd.to_datetime(df["DateTime"])

    seconds = np.zeros(len(df))
    #add seconds to start of the month or start of the day
    if start_month is True:
        for i, date in enumerate(df["DateTime"]):
            seconds[i] = date.second + 60 * date.minute + 3600 * date.hour + \
                         date.microsecond * 1e-6 + (date.day - 1) * 3600 * 24
    else:
        for i, date in enumerate(df["DateTime"]):
            seconds[i] = date.second + 60 * date.minute + 3600 * date.hour + \
                         date.microsecond * 1e-6


    df["Seconds"] = seconds

    #scale price to € cent
    df["Price"] = df["Price"] * 100

    return df

def match_orders(df, lst_index):
    # number of trades without a match
    n = 0
    for k,element in enumerate(lst_index):
        if element != []:
            # check if the random order was not preavously chosen
            flag = False
            # to avoid infinite loop repeat while at most 10 times
            i = 0
            while flag is False:
                trade = np.random.choice(element)
                if df["Type"].at[trade] != "Market" or i>10:
                    flag = True
                i += 1

            df["Type"].at[trade] = "Market"
            df["Sign"].at[trade] = -df["Sign"].at[trade]

        else:
            n += 1

    return df, n

def time(order_df, trade_df, time_interval):

    df = order_df.copy()
    lst_index = []

    for i in range(len(trade_df)):

        s = trade_df["Seconds"].iat[i]
        k = 1
        a = []

        while a == [] and k < time_interval:
            a = df[(df["Type"] == "Market/Cancel") & (df["Seconds"] > s - k) \
                 & (df["Seconds"] < s + k)].index.to_list()
            k += 1

        lst_index.append(a)

    df, n = match_orders(df, lst_index)
    # all remaining market/cancel are cancellations
    df["Type"].replace(["Market/Cancel"], "Cancel", inplace = True)

    return df, n

def time_price_volume_sign(order_df, trade_df, time_interval):
    df1 = order_df.copy()

    lst_index = []
    for i in range(len(trade_df)):
        p = trade_df["Price"].iat[i]
        v = trade_df["Volume"].iat[i]
        s = trade_df["Seconds"].iat[i]
        if trade_df["AggressorAction"].iat[i] == "Sell":
            sign = 1
        else:
            sign = -1

        k = 1
        a = []
        #verify if the broker is EEX in this case information on the sign is useless
        if trade_df["AggressorBroker"].iat[i] == "EEX":
            while a == [] and k < time_interval:
                a = df1[(df1["Volume"] == v) & (df1["Price"] == p) & (df1["Type"] == "Market/Cancel") \
                    & (df1["Seconds"] > s - k) & (df1["Seconds"] < s + k)].index.to_list()
                k += 1

        else:
            while a == [] and k < time_interval:
                a = df1[(df1["Volume"] == v) & (df1["Price"] == p) & (df1["Type"] == "Market/Cancel") \
                    & (df1["Seconds"] > s - k) & (df1["Seconds"] < s + k) & (df1["Sign"] == sign)].index.to_list()
                k += 1

        lst_index.append(a)

    df, n = match_orders(df1, lst_index)
    # all remaining market/cancel are cancellations
    df["Type"].replace(["Market/Cancel"], "Cancel", inplace = True)

    return df, n

def time_price_volume(order_df, trade_df, time_interval):

    df2 = order_df.copy()

    lst_index = []
    for i in range(len(trade_df)):
        p = trade_df["Price"].iat[i]
        v = trade_df["Volume"].iat[i]
        s = trade_df["Seconds"].iat[i]

        k = 1
        a = []
        #verify if the broker is EEX in this case information on the sign is useless
        while a == [] and k < time_interval:
            a = df2[(df2["Volume"] == v) & (df2["Price"] == p) & (df2["Type"] == "Market/Cancel") \
                & (df2["Seconds"] > s - k) & (df2["Seconds"] < s + k)].index.to_list()
            k += 1

        lst_index.append(a)

    df, n = match_orders(df2, lst_index)
    # all remaining market/cancel are cancellations
    df["Type"].replace(["Market/Cancel"], "Cancel", inplace = True)

    return df, n

def time_price(order_df, trade_df, time_interval):

    df3 = order_df.copy()
    lst_index = []

    for i in range(len(trade_df)):
        p = trade_df["Price"].iat[i]
        s = trade_df["Seconds"].iat[i]

        k = 1
        a = []
        #verify if the broker is EEX in this case information on the sign is useless
        while a == [] and k < time_interval:
            a = df3[(df3["Price"] == p) & (df3["Type"] == "Market/Cancel") \
                & (df3["Seconds"] > s - k) & (df3["Seconds"] < s + k)].index.to_list()
            k += 1

        lst_index.append(a)

    df, n = match_orders(df3, lst_index)
    # all remaining market/cancel are cancellations
    df["Type"].replace(["Market/Cancel"], "Cancel", inplace = True)

    return df, n

def time_volume(order_df, trade_df, time_interval):
    df4 = order_df.copy()

    lst_index = []
    for i in range(len(trade_df)):
        v = trade_df["Volume"].iat[i]
        s = trade_df["Seconds"].iat[i]

        k = 1
        a = []
        #verify if the broker is EEX in this case information on the sign is useless
        while a == [] and k < time_interval:
            a = df4[ (df4["Volume"] == v) & (df4["Type"] == "Market/Cancel") \
                & (df4["Seconds"] > s - k) & (df4["Seconds"] < s + k)].index.to_list()
            k += 1
        lst_index.append(a)

    df, n = match_orders(df4, lst_index)
    # all remaining market/cancel are cancellations
    df["Type"].replace(["Market/Cancel"], "Cancel", inplace = True)

    return df, n

def time_sign(order_df, trade_df, time_interval):

    df5 = order_df.copy()
    lst_index = []

    for i in range(len(trade_df)):

        v = trade_df["Volume"].iat[i]
        s = trade_df["Seconds"].iat[i]

        if trade_df["AggressorAction"].iat[i] == "Sell":
            sign = 1
        else:
            sign = -1

        k = 1
        a = []
        #verify if the broker is EEX in this case information on the sign is useless
        if trade_df["AggressorBroker"].iat[i] == "EEX":
            while a == [] and k < time_interval:
                a = df5[(df5["Type"] == "Market/Cancel") \
                    & (df5["Seconds"] > s - k) & (df5["Seconds"] < s + k)].index.to_list()
                k += 1
        else:
            while a == [] and k < time_interval:
                a = df5[(df5["Type"] == "Market/Cancel") & (df5["Seconds"] > s - k) \
                    & (df5["Seconds"] < s + k) & (df5["Sign"] == sign)].index.to_list()
                k += 1

        lst_index.append(a)

    df, n = match_orders(df5, lst_index)
    # all remaining market/cancel are cancellations
    df["Type"].replace(["Market/Cancel"], "Cancel", inplace = True)

    return df, n

def time_price_sign(order_df, trade_df, time_interval):

    df6 = order_df.copy()

    lst_index = []
    for i in range(len(trade_df)):
        p = trade_df["Price"].iat[i]
        s = trade_df["Seconds"].iat[i]

        if trade_df["AggressorAction"].iat[i] == "Sell":
            sign = 1
        else:
            sign = -1

        k = 1
        a = []
        #verify if the broker is EEX in this case information on the sign is useless
        if trade_df["AggressorBroker"].iat[i] == "EEX":
            while a == [] and k < time_interval:
                a = df6[(df6["Type"] == "Market/Cancel") & (df6["Price"] == p)\
                    & (df6["Seconds"] > s - k) & (df6["Seconds"] < s + k)].index.to_list()
                k += 1

        else:
            while a == [] and k < time_interval:
                a = df6[(df6["Type"] == "Market/Cancel") & (df6["Seconds"] > s - k) & (df6["Price"] == p) \
                    & (df6["Seconds"] < s + k) & (df6["Sign"] == sign)].index.to_list()
                k += 1

        lst_index.append(a)

    df, n = match_orders(df6, lst_index)
    # all remaining market/cancel are cancellations
    df["Type"].replace(["Market/Cancel"], "Cancel", inplace = True)

    return df, n

def time_volume_sign(order_df, trade_df, time_interval):

    df7 = order_df.copy()

    lst_index = []
    for i in range(len(trade_df)):
        v = trade_df["Volume"].iat[i]
        s = trade_df["Seconds"].iat[i]
        if trade_df["AggressorAction"].iat[i] == "Sell":
            sign = 1
        else:
            sign = -1

        k = 1
        a = []
        #verify if the broker is EEX in this case information on the sign is useless
        if trade_df["AggressorBroker"].iat[i] == "EEX":
            while a == [] and k < time_interval:
                a = df7[(df7["Type"] == "Market/Cancel") & (df7["Volume"] == v)\
                    & (df7["Seconds"] > s - k) & (df7["Seconds"] < s + k)].index.to_list()
                k += 1

        else:
            while a == [] and k < time_interval:
                a = df7[(df7["Type"] == "Market/Cancel") & (df7["Seconds"] > s - k) & (df7["Volume"] == v) \
                    & (df7["Seconds"] < s + k) & (df7["Sign"] == sign)].index.to_list()
                k += 1

        lst_index.append(a)

    df, n = match_orders(df7, lst_index)
    # all remaining market/cancel are cancellations
    df["Type"].replace(["Market/Cancel"], "Cancel", inplace = True)

    return df, n

def best_matching(order_df, trade_df, time_interval):

    df9 = order_df.copy()

    lst_index = []
    for i in range(len(trade_df)):
        p = trade_df["Price"].iat[i]
        v = trade_df["Volume"].iat[i]
        s = trade_df["Seconds"].iat[i]
        if trade_df["AggressorAction"].iat[i] == "Sell":
            sign = 1
        else:
            sign = -1

        k = 1
        a = []
        #verify if the broker is EEX in this case information on the sign is useless
        if trade_df["AggressorBroker"].iat[i] == "EEX":
            while a == [] and k < time_interval:
                a = df9[(df9["Volume"] == v) & (df9["Price"] == p) & (df9["Type"] == "Market/Cancel") \
                    & (df9["Seconds"] > s - k) & (df9["Seconds"] < s + k)].index.to_list()
                k += 1

            if a == []:
                k = 1
                while a == [] and k < time_interval:
                    a = df9[(df9["Volume"] == v) & (df9["Type"] == "Market/Cancel") \
                        & (df9["Seconds"] > s - k) & (df9["Seconds"] < s + k)].index.to_list()
                    k += 1
                if a == []:
                    k = 1
                    while a == [] and k < time_interval:
                        a = df9[(df9["Type"] == "Market/Cancel") \
                            & (df9["Seconds"] > s - k) & (df9["Seconds"] < s + k)].index.to_list()
                        k += 1
        else:
            while a == [] and k < time_interval:
                a = df9[(df9["Volume"] == v) & (df9["Price"] == p) & (df9["Type"] == "Market/Cancel") \
                    & (df9["Seconds"] > s - k) & (df9["Seconds"] < s + k) & (df9["Sign"] == sign)].index.to_list()
                k += 1

            if a == []:
                k = 1
                while a == [] and k < time_interval:
                    a = df9[(df9["Volume"] == v) & (df9["Price"] == p) & (df9["Type"] == "Market/Cancel") \
                        & (df9["Seconds"] > s - k) & (df9["Seconds"] < s + k)].index.to_list()
                    k += 1

                if a == []:
                    k = 1
                    while a == [] and k < time_interval:
                        a = df9[(df9["Volume"] == v) & (df9["Type"] == "Market/Cancel") \
                            & (df9["Seconds"] > s - k) & (df9["Seconds"] < s + k)].index.to_list()
                        k += 1

                    if a == []:
                        k = 1
                        while a == [] and k < time_interval:
                            a = df9[(df9["Type"] == "Market/Cancel") \
                                & (df9["Seconds"] > s - k) & (df9["Seconds"] < s + k)].index.to_list()
                            k += 1
        lst_index.append(a)

    df, n = match_orders(df9, lst_index)
    # all remaining market/cancel are cancellations
    df["Type"].replace(["Market/Cancel"], "Cancel", inplace = True)

    return df, n

def matching(order_df, trade_df, criterion = "time", time_interval = 5):

    switcher = {
    "time" : time,
    "time price volume sign": time_price_volume_sign,
    "time price volume": time_price_volume,
    "time price": time_price,
    "time volume": time_volume,
    "time sign": time_sign,
    "time price sign": time_price_sign,
    "time volume sign": time_volume_sign,
    "best matching": best_matching,
    }

    match_df, no_match = switcher[criterion](order_df, trade_df, time_interval)

    return match_df, no_match

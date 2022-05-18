from scipy.optimize import minimize
import numpy as np
import pandas as pd
import ZI
import os
from fbm import fgn

def distance_from_opposite(df):
    
    idx_buy = df[(df["Sign"] == 1) & (df["Type"] == "Limit")].index.to_numpy()
    idx_sell = df[(df["Sign"] == -1) & (df["Type"] == "Limit")].index.to_numpy()

    price_bid = df["Price"].loc[idx_buy[1:]].to_numpy()
    price_ask = df["Price"].loc[idx_sell[1:]].to_numpy()

    best_ask = df["AskPrice_0"].loc[idx_buy[1:] - 1].to_numpy()
    best_bid = df["BidPrice_0"].loc[idx_sell[1:] - 1].to_numpy()

    distance_buy = best_ask - price_bid
    distance_sell = price_ask - best_bid

    return distance_buy, distance_sell

def intensity_orders(df):

    mean_vol = df["TotVolume"].mean()

    mean_canc = df[df["Type"] == "Cancel"]["Volume"].mean()
    mean_limit = df[df["Type"] == "Limit"]["Volume"].mean()
    mean_market = df[df["Type"] == "Market"]["Volume"].mean()

    tot_time = 55800 - 37800

    rate_l = df[df["Type"] == "Limit"].shape[0] / tot_time
    rate_m = df[df["Type"] == "Market"].shape[0] / tot_time / mean_limit * mean_market
    rate_c = df[df["Type"] == "Cancel"].shape[0] / tot_time / mean_vol * mean_canc

    return rate_l, rate_m, rate_c

def load_data(dir_order, dir_message):

    #create a list of all the files in the folder
    lob_files = os.listdir(dir_order)
    lob_files.sort()
    lst_order = []
    for element in lob_files:
        # import data
        df = pd.read_csv(dir_order + element)
        df.fillna(0, inplace = True)

        # delete first two column and empty LOB
        df.drop(columns = ["Unnamed: 0", "key"], inplace = True)
        df.drop(df[df["AskPrice_0"] == 0].index.to_list(), inplace = True)
        df.drop(df[df["BidPrice_0"] == 0].index.to_list(), inplace = True)

        # scale price to dollar cent and add mid price and spread
        df["MidPrice"] = (df["BidPrice_0"] + df["AskPrice_0"]) / 2
        df["Spread"] = df["AskPrice_0"] - df["BidPrice_0"]

        # transform the column Datetime from string to datetime
        df["Datetime"]= pd.to_datetime(df["Datetime"])

        #create a new column that represent second to midnight
        seconds = np.zeros(len(df))
        for i, date in enumerate(df["Datetime"]):
            seconds[i] = date.second + 60 * date.minute + 3600 * date.hour + \
                                        date.microsecond * 1e-6
        df["Time"] = seconds


        df = df.loc[df["Datetime"].dt.day != 27]

        lst_order.append(df)

    clean_data = pd.concat(lst_order)
    clean_data.reset_index(inplace = True, drop = True)

    message = pd.read_csv(dir_message)

    message["DateTime"] = pd.to_datetime(message["DateTime"])
    data = pd.concat([clean_data, message[["Price", "Volume", "Sign", "Quote", "Type"]]], axis = 1)

    data = data.loc[data["Datetime"].dt.hour > 6]
    data = data.loc[data["Datetime"].dt.hour < 16]
    data = data.loc[data["Quote"] != "NoBest"]
    data = data.loc[data["Spread"] > 0]

    data.reset_index(inplace = True, drop = True)
    data.iloc[:,1:41:2] = data.iloc[:,1:41:2]*100
    data.loc[:,["Price", "Spread", "MidPrice"]] = data.loc[:,["Price", "Spread", "MidPrice"]]*100

    data["BuyVolume"] = data.iloc[:,2:40:4].sum(axis=1)
    data["SellVolume"] = data.iloc[:,4:44:4].sum(axis=1)
    data["TotVolume"] = data["BuyVolume"] + data["SellVolume"]

    return data

def truncated_power_law(x, a, s):
    ret = s * (a +1) / ((1+ s)**(a +1) - 1)
    ret = ret * (1 + s * x)**a
    return ret

def likelihood_tpl(params, p_index):
    a = params[0]
    s = params[1]

    N = len(p_index)
    ret = N * np.log(s * (a +1) / ((1+ s)**(a +1) - 1))
    ret += a * (np.log(1 + s * p_index)).sum()
    return -ret

def inverse_cdf_tpr(x,a,s):
    ret = ((1 + s)**(a + 1) - 1) * x + 1
    ret = (ret**(1 / (a + 1)) - 1) / s
    return ret

def compute_volume_index(df):
    # find indexes dataframe when a cancellation happened
    idx_buy = df.loc[(df["Type"] == "Cancel") & (df["Sign"] == 1)].index.to_numpy()
    idx_sell = df.loc[(df["Type"] == "Cancel") & (df["Sign"] == -1)].index.to_numpy()
    idx_buy = idx_buy[idx_buy > 0]
    idx_sell = idx_sell[idx_sell > 0]
    quote_buy = df["Quote"].loc[idx_buy]
    quote_sell = df["Quote"].loc[idx_sell]

    # create header list
    h_buy  = [f"BidVolume_{int(i)}" for i in range(10)]
    h_sell = [f"AskVolume_{int(i)}" for i in range(10)]

    # find the state of the LOB the moment before a cancellation occoured on the bid side
    volume_buy = df.loc[idx_buy - 1, h_buy]
    index_buy = np.zeros(volume_buy.shape[0])

    # for each cancellation find priority volume
    for k, i in enumerate(volume_buy.index.to_list()):
        header = [f"BidVolume_{int(j)}" for j in range(int(quote_buy.at[i+1]) + 1)]
        # find the number of orders ahead in the queue
        if len(header) > 1:
            index_buy[k] = volume_buy.loc[i,header[-1]] / 2 + volume_buy.loc[i,header[:-1]].sum()
        else:
            index_buy[k] = volume_buy.loc[i,header[-1]] / 2

    # repeat the same process for the ask side of the book
    volume_sell = df.loc[idx_sell - 1, h_sell]
    index_sell = np.zeros(volume_sell.shape[0])
    for k, i in enumerate(volume_sell.index.to_list()):
        header = [f"AskVolume_{int(j)}" for j in range(int(quote_sell.at[i+1]) + 1)]
        if len(header) > 1:
            index_sell[k] = volume_sell.loc[i,header[-1]] / 2 + volume_sell.loc[i,header[:-1]].sum()
        else:
            index_sell[k] = volume_sell.loc[i,header[-1]] / 2

    # to find the priority index divide the priority volume by the total volume
    # on the same side of the LOB
    p_buy = index_buy / df.loc[idx_buy - 1]["BuyVolume"].to_numpy()
    p_sell = index_sell / df.loc[idx_sell - 1]["SellVolume"].to_numpy()
    # merge the priority index for the bid and ask side in a unique array
    p_index = np.concatenate((p_buy, p_sell))

    return p_index

def compute_priority_index(df, guess):
    p_index = compute_volume_index(df)
    alpha, scale = minimize(likelihood_tpl, guess, bounds = ((-np.inf, 0), (0, None)), method='SLSQP').x
    print(minimize(likelihood_tpl, guess, bounds = ((-np.inf, 0), (0, None)), method='SLSQP'))
    return alpha, scale

def eexp(x, b0,b1,b2):
    return np.e**(b0 + b1 * np.log(x + 1) + b2 * (np.log(x+1))**2)

def double_eexp(x,y,k,a1,a2,b1,b2,ab):
    first  = k + b1 * np.log(x + 1) + b2 * (np.log(x+1))**2
    second = a1 * np.log(y + 1) + a2 * (np.log(y+1))**2
    interaction = ab * np.log(x + 1)*np.log(y + 1)
    return np.exp(first + second + interaction)

def logl_double_eexp(params, x1, x2, y1 ,y2, t):
    k = params[0]
    a1 = params[1]
    a2 = params[2]
    b1 = params[3]
    b2 = params[4]
    ab  = params[5]

    N = len(x1)
    spr = N*k + a1 * (np.log(x1 + 1)).sum() + a2 * ((np.log(x1+1))**2).sum()
    quote = b1 * (np.log(y1 + 1)).sum() + b2 * ((np.log(y1+1))**2).sum()
    interaction =  (ab * np.log(x1 + 1)*np.log(y1 + 1)).sum()

    spr_exp  = k + a1 * np.log(x2[1:] + 1) + a2 * (np.log(x2[1:] + 1))**2
    quote_exp = b1 * np.log(y2[1:] + 1) + b2 * (np.log(y2[1:] + 1))**2
    interaction_eexp = ab * np.log(x1 + 1)*np.log(y1 + 1)
    all_exp = np.exp(spr_exp + quote_exp + interaction_eexp)

    ret = spr + quote + interaction - (all_exp * (t[1:] - t[:-1])).sum()
    return -ret

def logl_eexp(params, x1, x2, t):
    b0 = params[0]
    b1 = params[1]
    b2 = params[2]
    N = len(x1)
    ret = N*b0 + b1 * (np.log(x1 + 1)).sum() + b2 * ((np.log(x1+1))**2).sum()
    ret -= (np.e**(b0 + b1 * np.log(x2[1:] + 1) + b2 * (np.log(x2[1:] + 1))**2) * (t[1:] - t[:-1])).sum()
    return -ret

def compute_intensity(df, column, guess, max_val = None, min_val = None, step = None):
    # compute time difference between 2 orders
    diff_time = df["Time"].diff().dropna().to_list()
    diff_time.append(0)
    df["DiffTime"] = diff_time

    # binning spread
    if max_val != None:
        df[column] = np.digitize(df[column], bins = np.arange(min_val,max_val,step))

    #compute intensity
    tot_time = df.groupby(column)["DiffTime"].sum()
    N = df.groupby(column)["DiffTime"].count()
    intensity = N / tot_time

    #find when there is a jump in the spread
    jumps = df[column].diff().fillna(0)
    idx = jumps[jumps != 0].index.to_list()

    time = df["Time"].loc[idx].to_numpy()
    q1 = df[column]
    q2 = df[column].loc[idx]

    # find params trough MLE
    print(minimize(logl_eexp, x0 = guess, args = (q1, q2, time), method= "SLSQP"))
    pars = minimize(logl_eexp, x0 = guess, args = (q1, q2, time), method= "SLSQP").x

    return pars, intensity, idx

def find_same_quote(df):
    same_quote = np.zeros(df.shape[0])
    for i in df.index.to_list():
        if df["Sign"].at[i] == 1:
            same_quote[i] = df["AskVolume_0"].at[i]
        else:
            same_quote[i] = df["BidVolume_0"].at[i]

    return same_quote

def find_same_quote_limit(df):
    same_quote = np.zeros(df.shape[0])
    for i in df.index.to_list():
        if df["Sign"].at[i] == 1:
            same_quote[i] = df["BuyVolume"].at[i]
        else:
            same_quote[i] = df["SellVolume"].at[i]

    return same_quote

def do_cancel_order(arr, sign, alpha = -1, scale = 1):

    n_orders_bid = arr[arr > 0].sum()
    n_orders_ask = -(arr[arr < 0].sum())

    #draw a random priority index from a truncated power law
    x = np.random.rand()
    priority_index = inverse_cdf_tpr(x, alpha, scale)

    # find the first order that has a priority index <= than the random one
    if sign == 1:
        arr_indexes = np.arange(0, n_orders_bid) / n_orders_bid
        pos = n_orders_bid - np.where(arr_indexes <= priority_index)[0][-1] - 1
    else:
        arr_indexes = np.arange(0, n_orders_ask) / n_orders_ask
        pos = np.where(arr_indexes <= priority_index)[0][-1] + n_orders_bid

    pos_orders = np.abs(arr).cumsum()
    price =  np.where(pos_orders > pos)[0][0]

    return price

def do_limit_order_opposite(arr, dist, lenght, sign):

    if sign == 1:
        best_price = np.where(arr > 0)[0][-1]
        opposite = np.where(arr < 0)[0][0]
        pos = 10e10
        while  pos >= opposite or pos < 0:
            pos = int(opposite - dist.rvs())


    else:
        best_price = np.where(arr < 0)[0][0]
        opposite = np.where(arr > 0)[0][-1]
        pos = 0
        while pos <= opposite or pos >= lenght:
            pos = int(opposite + dist.rvs())

    return pos

def do_market_order(arr, sign):

    if sign == 1:
        pos  = np.where(arr < 0)[0][0]
    else:
        pos = np.where(arr > 0)[0][-1]

    return pos

def sim(l_rate, m_rate, c_rate, k, iterations, distribution, burn, a, s, h_exp = None):

    #initialize LOB
    lob = np.ones(k, dtype = np.int16)
    lob[k//2:] = -1

    #compute sign using fractional gaussian noise
    if h_exp is None:
        arr_sign = np.random.choice([-1,1], size = int(iterations + burn))
    else:
        arr_sign = np.sign(fgn(n = int(iterations + burn), \
         hurst = h_exp,length = 1, method = 'daviesharte'))

    #initialize arrays
    spr = np.zeros(int(iterations + burn))
    mid_price = np.zeros(int(iterations + burn))
    arr_shift = np.zeros(int(iterations + burn))
    arr_type = np.zeros(int(iterations + burn))
    tot_orders = np.zeros(int(iterations + burn))

    for i in range(int(iterations + burn)):
        bid_size = lob[lob > 0].sum()
        ask_size = -lob[lob < 0].sum()

        sign = arr_sign[i]

        tot = l_rate + m_rate + c_rate * np.abs(lob).sum()
        # find type next order and make sure to not cancel the last order in the
        # bid or ask side of the book
        FLAG = False
        while FLAG is False:
            o_type = np.random.choice([0,1,2], p = [l_rate / tot, m_rate / tot, c_rate * np.abs(lob).sum() / tot])

            if bid_size > 1 and ask_size > 1:
                FLAG = True

            elif bid_size == 1 and sign == 1 and o_type == 2:
                FLAG= False

            elif bid_size == 1 and sign == -1 and o_type == 1:
                FLAG= False

            elif ask_size == 1 and  sign == -1 and o_type == 2:
                FLAG= False

            elif ask_size == 1 and sign == 1 and o_type == 1:
                FLAG= False

            else:
                FLAG = True

        mp = ZI.find_mid_price(lob)

        if o_type == 0:
            price = do_limit_order_opposite(lob, distribution, k, sign)

        elif o_type == 1:
            price = do_market_order(lob, sign)

        else:
            price = do_cancel_order(lob, sign, a, s)
            sign = - sign


        lob[price] += sign
        spr[i] = ZI.find_spread(lob)
        new_mp = ZI.find_mid_price(lob)
        mid_price[i] = new_mp
        arr_type[i] = o_type
        tot_orders[i] = np.abs(lob).sum()

        shift = int(new_mp + 0.5 - k//2)
        arr_shift[i] = shift

        if shift > 0:
            num_orders = lob[:shift].sum()
            lob[:-shift] = lob[shift:]
            lob[-shift:] = np.zeros(len(lob[-shift:]))
            lob[0] += num_orders

        elif shift < 0:
            num_orders = lob[shift:].sum()
            lob[-shift:] = lob[:shift]
            lob[:-shift] = np.zeros(len(lob[:-shift]))
            lob[-1] += num_orders

    price = arr_shift.cumsum() + mid_price
    price -= price[-iterations]

    return lob[-iterations:], spr[-iterations:], price[-iterations:], arr_type[-iterations:]

def double_eexp(x,y,k,a1,a2,b0,b1,b2,ab):
    first  = k + b1 * np.log(x + 1) + b2 * (np.log(x+1))**2
    second = a1 * np.log(y + 1) + a2 * (np.log(y+1))**2
    interaction = ab * np.log(x + 1)*np.log(y + 1)
    return np.exp(first + second + interaction)

def logl_double_eexp(params, x1, x2, y1 ,y2, t):
    k = params[0]
    a1 = params[1]
    a2 = params[2]
    b1 = params[3]
    b2 = params[4]
    ab  = params[5]

    N = len(x1)
    spr = N*k + a1 * (np.log(x1 + 1)).sum() + a2 * ((np.log(x1+1))**2).sum()
    quote = b1 * (np.log(y1 + 1)).sum() + b2 * ((np.log(y1+1))**2).sum()
    interaction =  (ab * np.log(x1 + 1)*np.log(y1 + 1)).sum()

    spr_exp  = k + a1 * np.log(x2[1:] + 1) + a2 * (np.log(x2[1:] + 1))**2
    quote_exp = b1 * np.log(y2[1:] + 1) + b2 * (np.log(y2[1:] + 1))**2
    interaction_eexp = ab * np.log(x2[1:] + 1)*np.log(y2[1:] + 1)
    all_exp = np.exp(spr_exp + quote_exp + interaction_eexp)

    ret = spr + quote + interaction - (all_exp * (t[1:] - t[:-1])).sum()
    return -ret

def double_intensity(df, idx_spr, idx_q, guess, max_val = None, min_val = None, step = None):
    # compute time difference between 2 orders
    diff_time = df["Time"].diff().dropna().to_list()
    diff_time.append(0)
    df["DiffTime"] = diff_time

    # binning spread
    if max_val != None:
        df["Spread"] = np.digitize(df["Spread"], bins = np.arange(min_val,max_val,step))

    market_idx = idx_spr + idx_q
    market_idx.sort()
    time = df["Time"].loc[market_idx].to_numpy()
    s1 = df["Spread"]
    s2 = df["Spread"].loc[market_idx]
    q1 = df["SameQuote"]
    q2 = df["SameQuote"].loc[market_idx]

    # find params trough MLE
    print(minimize(logl_double_eexp, x0 = guess, args = (s1, s2, q1, q2, time), method= "SLSQP"))
    pars = minimize(logl_double_eexp, x0 = guess, args = (s1, s2, q1, q2, time),method= "SLSQP").x

    return pars

import numpy as np
import pandas as pd
import scipy

def compute_weight(new_df, idx, distance):
    ss = pd.unique(new_df["Spread"].loc[idx])
    weight = np.zeros(distance.shape[0])

    for j, val in enumerate(distance):
        if ss[ss > val].shape[0] != 0:
            weight[j] = ss.shape[0] / ss[ss > val].shape[0]

    return weight

def resample_data(distance, weight, N):
    data = pd.DataFrame(distance, columns= ["Dist"])
    data["Weight"] = weight
    resampled = data["Dist"].sample(N, weights = data["Weight"],
                                    replace = True).to_numpy()
    return resampled

def weighted_distance(new_df, n_resample = 200_000, condition = None):
    #Find index limit orders on the bid side
    if condition is None:
        bid_orders = new_df[(new_df["Sign"] == 1) \
                            & (new_df["Type"] == "Limit")].index.to_numpy()
    else:
        bid_orders = new_df[(new_df["Sign"] == 1) \
                            & (new_df["Type"] == "Limit") & condition].index.to_numpy()

    bid_orders[0] = 2

    #Compute distance from the same side best price
    best_bid  = new_df["BidPrice_0"].loc[bid_orders - 1].to_numpy()
    bid_price = new_df["Price"].loc[bid_orders].to_numpy()
    distance_bid =  bid_price - best_bid

    #Compute weighted distance and resample data according to the weight
    weight_bid = compute_weight(new_df, bid_orders - 1, distance_bid)
    resampled_bid = resample_data(distance_bid, weight_bid, n_resample)

    #Find index limit orders on the ask side
    if condition is None:
        ask_orders = new_df[(new_df["Sign"] == -1) \
                            & (new_df["Type"] == "Limit")].index.to_numpy()
    else:
        ask_orders = new_df[(new_df["Sign"] == -1) \
                            & (new_df["Type"] == "Limit") & condition].index.to_numpy()
    ask_orders[0] = 2

    #Compute distance from the same side best price
    best_ask = new_df["AskPrice_0"].loc[ask_orders - 1].to_numpy()
    ask_price =  new_df["Price"].loc[ask_orders].to_numpy()
    distance_ask = best_ask - ask_price

    #Compute weighted distance and resample data according to the weight
    weight_ask = compute_weight(new_df, ask_orders - 1, distance_ask)
    resampled_ask = resample_data(distance_ask, weight_ask, n_resample)

    return resampled_bid, resampled_ask

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

def liquidity(x, rate_m, rate_l, vol_m, vol_l, Q):
    v = rate_l / x
    d = rate_m / x
    q = vol_m / vol_l
    num = d * q**(v/(1-q))
    den = scipy.special.hyp2f1(d, - v / (1-q), 1 + d, 1 - q)
    return vol_m * (v / q - d + num / den) - Q

class double_t(scipy.stats.rv_continuous):
    def __init__(self, df1, df2, mu1, mu2, std1, std2, p):
        super().__init__()
        self.df1 = df1
        self.df2 = df2
        self.mu1 = mu1
        self.mu2 = mu2
        self.std1 = std1
        self.std2 = std2
        self.p = p
        pass

    def _pdf(self, x):
        ret = (1 - self.p) * scipy.stats.t.pdf(x, self.df1, self.mu1, self.std1)
        ret += self.p * scipy.stats.t.pdf(x, self.df2, self.mu2, self.std2)
        return ret

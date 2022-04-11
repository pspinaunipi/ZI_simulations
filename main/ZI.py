import numpy as np
from numba import njit
import matplotlib.pyplot as plt

@njit()
def inter_arrival(tau):
    n = np.random.random()
    arrival = -np.log(1 - n) / tau
    return arrival

@njit()
def find_mid_price(arr):
    best_bid = np.where(arr > 0)[0][-1]
    best_ask = np.where(arr < 0)[0][0]

    return (best_bid + best_ask) / 2

@njit()
def rand_sign():
    array = np.array([-1,1])
    return int(np.random.choice(array))

@njit()
def do_limit_order(mid_p, kk):

    pos = np.random.randint(kk)
    if pos < mid_p:
        sign = +1
    elif pos > mid_p:
        sign = -1
    else:
        sign = rand_sign()
    return pos, sign

@njit()
def do_market_order(arr):

    sign = rand_sign()
    if sign == 1:
        pos  = np.where(arr < 0)[0][0]
    else:
        pos = np.where(arr > 0)[0][-1]

    return pos, sign

@njit()
def do_cancel_order(arr, mid_p):

    n_orders = np.abs(arr).sum()
    pos = np.random.randint(n_orders)

    pos_orders = np.abs(arr).cumsum()

    price =  np.where(pos_orders > pos)[0][0]

    if arr[price] > 0:
        sign = -1
    else:
        sign = 1

    return price, sign

@njit()
def find_spread(arr):
    best_bid = np.where(arr > 0)[0][-1]
    best_ask = np.where(arr < 0)[0][0]

    return best_ask - best_bid

@njit()
def sim_LOB(l_rate, m_rate, c_rate, k = 100, iterations = 10_000, all_lob = False):

    #initialize LOB
    lob = np.ones(k, dtype = np.int16)
    lob[k//2:] = -1

    spr = np.zeros(iterations)
    mid_price = np.zeros(iterations)
    arr_shift = np.zeros(iterations)

    #compute inter arrival times
    time_l = inter_arrival(k * l_rate)
    time_m = inter_arrival(2 * m_rate)
    time_c = inter_arrival(c_rate * np.abs(lob).sum())
    times = np.array([time_l, time_m, time_c])


    all = []

    for i in range(iterations):
        # find type next order
        o_type = np.argmin(times)
        mp = find_mid_price(lob)

        if o_type == 0:
            price, sign = do_limit_order(mp, k)
            #update_times
            times -= times[o_type]
            times[o_type] = inter_arrival(k * l_rate)

        elif o_type == 1:
            price, sign = do_market_order(lob)
            #update_times
            times -= times[o_type]
            times[o_type] = inter_arrival(2 * m_rate)

        else:
            price, sign = do_cancel_order(lob, mp)
            #update_times
            times -= times[o_type]
            times[o_type] = inter_arrival(c_rate * np.abs(lob).sum())

        #update lob spread and mid price
        lob[price] += sign
        spr[i] = find_spread(lob)
        new_mp = find_mid_price(lob)
        mid_price[i] = new_mp - k // 2
        shift = int(new_mp - k//2)
        arr_shift[i] = shift

        if all_lob is True:
            all.append(lob.copy())

        #center LOB around mid price
        if shift > 0:
            lob[:-shift] = lob[shift:]
            lob[-shift:] = np.zeros(len(lob[-shift:]))
        elif shift < 0:
            lob[-shift:] = lob[:shift]
            lob[:-shift] = np.zeros(len(lob[:-shift]))

    price = arr_shift.cumsum() + mid_price

    return lob, spr, price, all

def estimate_parameters(X_lo, X_mo, X_c, X_lo_sp, V):

    N_lo = len(X_lo)
    N_mo = len(X_mo)
    N_c  = len(X_c)

    tot  = N_mo + N_lo + N_c

    v0 = X_lo.mean()

    u  = 0.5 / tot * X_mo.sum() / v0
    v  = 0.5 / tot * X_c.sum() / V
    l_all  = 0.5 * N_lo / tot
    n = 2 * (1 + ((X_lo_sp // 2).mean()))
    l = l_all / n

    return l, u, v

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
        mid_price[i] = new_mp
        shift = int(new_mp - k//2)
        arr_shift[i] = shift

        if all_lob is True:
            all.append(lob)

        #center LOB around mid price
        if shift > 0:
            lob[:-shift] = lob[shift:]
            lob[-shift:] = np.zeros(len(lob[-shift:]))
        elif shift < 0:
            lob[-shift:] = lob[:shift]
            lob[:-shift] = np.zeros(len(lob[:-shift]))

    price = arr_shift.cumsum() + mid_price


    return lob, spr, price, all


if __name__ == "__main__":
    rate_lim = 0.023
    rate_m   = 0.062
    rate_del = 0.109

    llob, sp, mm = sim_LOB(rate_lim, rate_m, rate_del, 250, 300_000, all_lob = False)
    plt.bar(np.arange(250), gg[-500])
    plt.show()
    vol = np.sqrt(((mm[1:]- mm[:-1])**2).mean())
    print(sp.mean())
    print(vol)
    plt.plot(mm)
    plt.show()

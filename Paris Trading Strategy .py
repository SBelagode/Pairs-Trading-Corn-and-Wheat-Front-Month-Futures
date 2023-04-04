#!/usr/bin/env python
# coding: utf-8

# The intuition for this idea is based off a research paper that I read that mentioned that there appear to be interesting short-term correlations between various commodity futures. However, the idea to pairstrade, and all other ideas are my own. 
# 
# The paper link: https://www.hindawi.com/journals/complexity/2021/8848424/
# 
# This trade is based on the idea to pairs trade corn and wheat, using front end rolling futures contracts that I downloaded from marketwatch. My idea was to only trade when the log difference in price (the price difference is defined as: log(wheat) - slope * log(corn)) was above (in absolute value) the entry z score until the log difference was below (in absolute value) the exit z score. When I first backtested this, I defined the spread as log(wheat) - log (corn). This is missing the slope term. My initial idea was also motivated by an EPU index value (Economic Policy Uncertainty).
# 
# Trade expression: 
# 
# I am basing my trades off large divergences from short-term sma of the log
# spread between two commodities, having an entry z-score based on how many standard
# deviations the current day’s spread is from the commodity spread sma mean. I would then hold
# until I reach my exit z-score, which would be fewer standard deviations away from the sma price
# difference mean as I believe that the ratios between the commodities is mean reverting in the
# short run. Based on my entry z-score, I would short the overvalued commodity, and long the
# undervalued commodity. However, the additional filter I would place on this trade is that I would
# only initiate the trade over a certain EPU threshold value, as the EPU value may affect the
# cointegration between the commodities.
# 

# In[ ]:


## importing appropriate libraries and gathering data from my computer to create a data set to work with 
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as ts
from scipy.stats import linregress


period = ('2011-01-03', '2016-12-30')
days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

data = pd.read_csv("/Users/vinayabelagode/Desktop/Pairstrade backtest/PriceData.csv")
data = data[::-1]
for i in range(len(data)):
    if data['Date'][i][1] == "/":
        data['Date'][i] = '0' + data['Date'][i]
    if data['Date'][i][4] == "/":
        data['Date'][i] = data['Date'][i][:3] + '0' + data['Date'][i][3:]
    data['Date'][i] = '20' + data['Date'][i][6:] + '-' + data['Date'][i][:2] + '-' + data['Date'][i][3:5]
data.set_index("Date", drop = False, inplace = True)

## ask about .set_index

#print(data)

for key in data.keys():
    if key == 'EPU':
        continue
    for i in range(len(data)):
        if isinstance(data[key][i], str) and data[key][i][1] == ',':
                data[key][i] = "1" + data[key][i][2:]


data['Close Wheat'] = pd.to_numeric(data['Close Wheat'])
data['Close Soybean'] = pd.to_numeric(data['Close Soybean'])

print(data)


# In[ ]:


## cointegration test: This is the first step in identifying whether pairs trading is viable between two financial time
#series data
# cointegration tests for stationarity between two time series. Essentially, I'm trying to see if there is some 
#quantifiable spread that I can predict the two commodities will repeatedly revert back to. If so, I can then 
#trade in period where the current spread diverges from this usual spread, since I am hypothesizing that 
#the current spread will eventually revert back to this usual spread.

def plot_pairs(data1, data2):
    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle('Corn and Wheat')
    ax1.plot(data1)
    ax2.plot(data2)
    plt.show()
def scatter_plot(data1, data2):
    plt.scatter(data1.values, data2.values)
    plt.suptitle('corn vs wheat')
    plt.xlabel('Corn')
    plt.ylabel('Wheat')
    plt.show()
if True:
    pair1 = data['Close Corn ']['2011-01-03':'2016-12-30']
    pair2 = data['Close Wheat']['2011-01-03':'2016-12-30']
    plot_pairs(pair1, pair2)
    scatter_plot(pair1, pair2)
    # Linear Regression
    result = linregress(pair1.values[:], pair2.values[:])
    print(result)
    # Create the residual series
    residuals = pair1 - result.slope * pair2
    adf = ts.adfuller(residuals)
    print(adf)
    
  
# replicating Cointegration Test if found online
pair1 = data['Close Corn ']['2011-01-03':'2016-12-30']
pair2 = data['Close Wheat']['2011-01-03':'2016-12-30']
CornP = pair1.values
WheatP = pair2.values
coint = ts.coint(CornP, WheatP)
print('Cointegration between Corn Prices and Wheat Prices', coint)


# In[ ]:


## visualizing the spread 

spread = []
for i in data['Date']['2011-01-03':'2016-12-30']: 
    add = np.log(data['Close Wheat'][i]) - 1.047 * np.log(data['Close Corn '][i])
    spread.append(add)

plt.plot(spread)
plt.show()
print('mean', np.mean(spread))



# Now, I will be optimizing, over my training set data from Jan 2011 to December 2016, to see what trade set up I should use. I first create various useful functions that I need. 
# 
# For instance, my rolling_regression_in_sample function is used to when I define spread as log(wheat) - slope * log(corn), as research into pairs trading suggested I should define spread in this way. The slope here represents the slope from regressing Corn on Wheat. Using log ratios would make sense as the ratios between two stocks would be lognormally distributed since these ratios cannot trade below zero and have no upper bound. B represents the regression coefficient when regressing wheat and corn prices. Again, it represents how much of stock 2 you
# should trade per share of stock1. The slope value is what this function calculates. 
# 
# My sma_epu_in_sample function calculates the simple moving average value of my EPU (Economic Policy Uncertainty) index data for a certain number of lookback days. I need this value because I want to calculate a threshold value. This threshold value will be used to determine whether or not I trade under/over this value. 1. The length of the lookback period for which this simple moving average will be based on and 2. the fact of whether I trade under or over this value are the 2 things I will be optimizing for. 
# 
# My sma_diff_in_sample and commodity_difference functions are used to track the difference (otherwise known as spread) between my commodity prices. 
# 
# So, as you will see in the code, I optimized for the simple moving average (sma) lookback period
# (how many days’ worth of data I will consider for my calculations for this particular sma) for
# EPU, the simple moving average (sma) lookback period for rolling regression, the simple
# moving average (sma) lookback period for spread, the entry z score, and the exit z score.
# 
# To optimize over lookback period for the sma, entry z-score, and exit zscore, I used for loops to
# try all combinations with lookback starting at 20 – 180 days with a 20 day step, 0.2 to 3 with a
# 0.1 step for entry z score, 0.1 to 2.9 with a 0.1 step for exit z score, EPU lookbacks starting at
# 180 days to 360 days with 90 day steps, and Rolling Regression lookbacks starting at 90 days to
# 390 days with 30 day steps.
# 
# Note: Sharpe ratio is a metric to measure risk-adjusted returns.
# 

# In[ ]:


## Here, on my in sample data, I am optimizing for the best set up that will return the highest Sharpe ratio

## optimizing over values
import sklearn

from sklearn.linear_model import LinearRegression
def rolling_regression_in_sample(lookback_days, r):
    window_size = lookback_days
    x1 = r 
    list_for_sma = data['Date']
    index_of_start_period = list_for_sma.index.get_loc('2011-01-03')
    index_of_start_of_window = index_of_start_period - window_size
    value_of_start_of_window = str(data['Date'][index_of_start_of_window])
    y = data['Close Corn '][value_of_start_of_window:'2016-12-30'][x1 : x1 + window_size]
    x = data['Close Wheat'][value_of_start_of_window:'2016-12-30'][x1 : x1 + window_size]
    x = np.array(x)
    x = x.reshape((-1, 1))

    model = LinearRegression().fit(x, y)

    slope = model.coef_
    return slope 
def sma_epu_in_sample(lookback_days, y): 
    #store lookback period size for sma 
    window_size = lookback_days
    
    x = y 
    list_for_sma = data['Date']
    index_of_start_period = list_for_sma.index.get_loc('2011-01-03')
    index_of_start_of_window = index_of_start_period - window_size
    value_of_start_of_window = str(data['Date'][index_of_start_of_window])
    window_epu = data['EPU'][value_of_start_of_window:'2016-12-30'][x : x + window_size]
    
    window_ratio = []
    for i in range(len(window_epu)): 
        window_ratio.append(window_epu[i])
    return window_ratio

def sma_diff_in_sample(lookback_days, y, slope): 
    #store lookback period size for sma 
    window_size = lookback_days
    x = y 
    list_for_sma = data['Date']
    index_of_start_period = list_for_sma.index.get_loc('2011-01-03')
    index_of_start_of_window = index_of_start_period - window_size
    value_of_start_of_window = str(data['Date'][index_of_start_of_window])
    window_cp = data['Close Corn '][value_of_start_of_window:'2016-12-30'][x : x + window_size]
    window_wp = data['Close Wheat'][value_of_start_of_window:'2016-12-30'][x : x + window_size]
    window_ratio = []
    for i in range(len(window_cp)): 
        window_ratio.append(np.log(window_wp[i]) - slope * np.log(float(window_cp[i])))
        
    return window_ratio

def commodity_difference(i, slope): 
    diff = np.log(data['Close Wheat'][i]) - slope * np.log(float(data['Close Corn '][i]))
    return diff





def corn_wheat_optimization_setup_1():
    #creating data frame to store configurations and results
    dfSharpes = pd.DataFrame({'Commodity Lookback':0, 'Entry Z-Score':0, 'Exit Z-Score':0, 'Sharpe': 0, 'Trades': 0, 'ELookback':0, 'Rolling Regress Lookback':0}, index = [0])
    
    
    #iterating through various configuraitons 
    for EPU_lookback in np.arange(180, 400, 90):
        for rolling_regress_lookback in np.arange(150, 400, 30): 
            for lookback_days in np.arange(20,180,20): 
                for entry_zscore in np.arange(1, 3, 0.1): 
                    for exit_zscore in np.arange(1, 2.9, 0.1):
                        if (entry_zscore > (exit_zscore + 0.1)): 
                            
                            
                            #initializing important values 
                            
                            pairs_trade = 0 #0 = nothing, 1 = long first short second, -1 = long second short first
                            y = 0
                            trade_pnl = []
                            date_per_trade_pnl = []
                            f = 0
                            trade_number = 0 
                            r = 0 



                            ## now loop through test interval 
                            for i in data['Date']['2011-01-03':'2016-12-30']:
                                
                                #calculating needed values
                                epu_threshold = np.median(sma_epu_in_sample(EPU_lookback, f))
                                f += 1
                                
                                slope = rolling_regression_in_sample(rolling_regress_lookback, r)
                                r += 1
                                
                                ratio_ma = sma_diff_in_sample(lookback_days, y, slope)
                                y += 1

                                


                                #seeing if I should enter a trade or not
                                if pairs_trade == 0:
                                    
                                    
                                    #threshold condition
                                    if data['EPU'][i] > epu_threshold:

                                        today_spread = commodity_difference(i, slope)
                                        
                                        
                                        #seeing if divergence is enough to enter trade
                                        if (today_spread - np.mean(ratio_ma))/ np.std(ratio_ma) > entry_zscore:

                                            pairs_trade = 1
                                            #Sell wheat and Buy corn
                                            initial_corn_price = data['Close Corn '][i] * slope
                                            initial_wheat_price = float(data['Close Wheat'][i])
                                            trade_ratio = np.mean(ratio_ma)
                                            trade_std = np.std(ratio_ma)
                                            trade_slope = slope

                                            date_entered = i


                                        #seeing if divergence on other end is enough to enter trade
                                        elif (today_spread - np.mean(ratio_ma))/ np.std(ratio_ma) < -entry_zscore:

                                            pairs_trade = -1
                                            # Buy Wheat and Sell Corn 
                                            initial_corn_price = data['Close Corn '][i] * slope
                                            initial_wheat_price = float(data['Close Wheat'][i])
                                            trade_ratio = np.mean(ratio_ma)
                                            trade_std = np.std(ratio_ma)
                                            trade_slope = slope

                                            date_entered = i


                                #if entered trade by selling wheat and buying corn because I believe spread is 
                                #too high
                                if pairs_trade == 1:

                                    today_spread = commodity_difference(i, trade_slope)

                                    final_corn_price = data['Close Corn '][i] * trade_slope
                                    final_wheat_price = float(data['Close Wheat'][i]) 
                                    #PnL from trade recalculated every day 
                                    PnL = -((final_wheat_price - initial_wheat_price)/initial_wheat_price) + ((final_corn_price - initial_corn_price)/initial_corn_price)



                                    trade_pnl.append(PnL)
                                    today_date = i
                                    date_per_trade_pnl.append(today_date)

                                    initial_corn_price = data['Close Corn '][i] * trade_slope
                                    initial_wheat_price = float(data['Close Wheat'][i])
                                    
                                    #seeing if exit conditions are met
                                    if (today_spread - trade_ratio)/ trade_std < exit_zscore:


                                        date_exited = i
                                        #buy back wheat and Sell corn 

                                        pairs_trade = 0

                                        trade_number += 1


                                    #stop loss 
                                    if PnL <= -20:

                                        date_exited = i

                                        pairs_trade = 0

                                        trade_number += 1



                                # if I am selling buying wheat and selling corn because I believe the spread 
                                #is too small
                                elif pairs_trade == -1:

                                    today_spread = commodity_difference(i, trade_slope)
                                    final_corn_price = data['Close Corn '][i] * trade_slope
                                    final_wheat_price = float(data['Close Wheat'][i]) 
                                    #PnL from trade recalculated every day 
                                    PnL = -((final_corn_price - initial_corn_price)/initial_corn_price) + \
                                    ((final_wheat_price - initial_wheat_price)/initial_wheat_price)

                                    trade_pnl.append(PnL)


                                    initial_corn_price = data['Close Corn '][i] * trade_slope
                                    initial_wheat_price = float(data['Close Wheat'][i])
                                    
                                    #seeing if exit conditions are met
                                    if (today_spread - trade_ratio)/ trade_std > -exit_zscore:

                                        date_exited = i

                                        pairs_trade = 0

                                        trade_number += 1


                                    #stop loss
                                    if PnL <= -20: 

                                        date_exited = i

                                        pairs_trade = 0

                                        trade_number += 1




                            #calculate sharpe ratio 
                            #average of PnL
                            avg_pnl = np.mean(trade_pnl)
                            #sharpe 
                            sharpe_before_adj = avg_pnl/np.std(trade_pnl)
                            sharpe = sharpe_before_adj * math.sqrt(252)
                            new_row = pd.Series({'Commodity Lookback': lookback_days, 'Entry Z-Score': entry_zscore,'Exit Z-Score': exit_zscore, 'Sharpe': sharpe, 'Trades': trade_number, 'ELookback': EPU_lookback, 'Rolling Regress Lookback':rolling_regress_lookback})


                            dfSharpes = pd.concat([dfSharpes, new_row.to_frame().T], ignore_index = True)

    return dfSharpes








data_setup_1 = corn_wheat_optimization_setup_1()
data_setup_1.sort_values(by=['Sharpe'], ascending=False, inplace=True)
print(data_setup_1.head(50))


# In[ ]:


#now only selecting rows where entry z score is larger than exit z score and with at least 30 trades so that 
#I can lower variance by only selecting from trades that have repeatedly proven to be profitable (as opposed to 
#selecting a trading idea that only traded <5 times)


#creating new data frame to store data
select_rows_setup_1 = pd.DataFrame({'Commodity Lookback':0, 'Entry Z-Score':0, 'Exit Z-Score':0, 
                                    'Sharpe': 0, 'Trades': 0, 'ELookback':0, 'Rolling Regress Lookback':0},
                                   index = [0])


#iterating through backtest optimization data 
for i in range(len(data_setup_1)): 
    
    if (data_setup_1.loc[i][1] > (data_setup_1.loc[i][2] + 0.1)) and (data_setup_1.loc[i][4] > 30): 
        new_row_setup_1 = pd.Series({'Commodity Lookback': data_setup_1.loc[i][0], 'Entry Z-Score':
                                     data_setup_1.loc[i][1],'Exit Z-Score': data_setup_1.loc[i][2], 
                                     'Sharpe': data_setup_1.loc[i][3], 'Trades': data_setup_1.loc[i][4], 
                                     'ELookback': data_setup_1.loc[i][5], 'Rolling Regress Lookback':
                                     data_setup_1.loc[i][6]})
        select_rows_setup_1 = pd.concat([select_rows_setup_1, new_row_setup_1.to_frame().T], ignore_index = True)

        

select_rows_setup_1.sort_values(by=['Trades'], ascending=False, inplace=True)
select_rows_setup_1 = select_rows_setup_1[:-1]
print(select_rows_setup_1.head(30))


# In[ ]:


#organizing by best Sharpe


select_rows_setup_1.sort_values(by=['Sharpe'], ascending=False, inplace=True)
select_rows_setup_1 = select_rows_setup_1[:-1]
print(select_rows_setup_1.head(30))


# After obtaining the best performing results, I decide to apply the same trade in my out of sample dat from Jan 
# 2017 to March 2020 and plot the results. 
# 
# I had an entry z score of 2.4, entry z score of 2.2 (it’s important to
# remember these are absolute values in the sense that I would enter the trade if my spread had a z
# score lower than -2.4 or, in other words, was 2.4 standard deviations below the mean),
# commodity spread lookback period of 20 days, EPU lookback period of 360 days, and Rolling
# Regression period of 180 days.

# In[ ]:


def subtract_dates(date_enter, date_exit): 
    date1 = str(date_enter)
    date2 = str(date_exit)
    date_column = data['Date']
    difference = date_column.index.get_loc(date2) - date_column.index.get_loc(date1)
    ## returns how many TRADING days are in between the date I enter a trade and the date I exit a trade 
    return difference 

## testing best sharpe results from most trades (trade 1775 in the above kernel) results on out of sample 


from sklearn.linear_model import LinearRegression
def rolling_regression_out_sample(lookback_days, r):
    window_size = lookback_days
    x1 = r 
    list_for_sma = data['Date']
    index_of_start_period = list_for_sma.index.get_loc('2017-01-03')
    index_of_start_of_window = index_of_start_period - window_size
    value_of_start_of_window = str(data['Date'][index_of_start_of_window])
    y = data['Close Corn '][value_of_start_of_window:'2020-03-06'][x1 : x1 + window_size]
    x = data['Close Wheat'][value_of_start_of_window:'2020-03-06'][x1 : x1 + window_size]
    x = np.array(x)
    x = x.reshape((-1, 1))

    model = LinearRegression().fit(x, y)

    slope = model.coef_
    return slope 
def sma_epu_out_sample(lookback_days, y): 
    #store lookback period size for sma 
    window_size = lookback_days
    
    x = y 
    list_for_sma = data['Date']
    index_of_start_period = list_for_sma.index.get_loc('2017-01-03')
    index_of_start_of_window = index_of_start_period - window_size
    value_of_start_of_window = str(data['Date'][index_of_start_of_window])
    window_epu = data['EPU'][value_of_start_of_window:'2020-03-06'][x : x + window_size]
    
    window_ratio = []
    for i in range(len(window_epu)): 
        window_ratio.append(window_epu[i])
    return window_ratio

def sma_diff_out_sample(lookback_days, y, slope): 
    #store lookback period size for sma 
    window_size = lookback_days
    x = y 
    list_for_sma = data['Date']
    index_of_start_period = list_for_sma.index.get_loc('2017-01-03')
    index_of_start_of_window = index_of_start_period - window_size
    value_of_start_of_window = str(data['Date'][index_of_start_of_window])
    window_cp = data['Close Corn '][value_of_start_of_window:'2020-03-06'][x : x + window_size]
    window_wp = data['Close Wheat'][value_of_start_of_window:'2020-03-06'][x : x + window_size]
    window_ratio = []
    for i in range(len(window_cp)): 
        window_ratio.append(np.log(window_wp[i]) - slope * np.log(float(window_cp[i])))
        
    return window_ratio

def calc_sharpe_corn_wheat_diff_out_sample(entry_zscore, exit_zscore, lookback_days, EPU_lookback, reg_lookback_days):
    
    
   
    
    if entry_zscore > exit_zscore: 

        pairs_trade = 0 #0 = nothing, 1 = long first short second, -1 = long second short first
        y = 0
        trade_pnl = []
        date_per_trade_pnl = []
        
        trade_number = 0
        trade_stop_loss = 0
        days_in_trade = []
        f = 0
        r = 0



        ## now loop through test interval 
        for i in data['Date']['2017-01-03':'2020-03-06']:
            epu_threshold = np.median(sma_epu_out_sample(EPU_lookback, f))
            f += 1
            
            slope = rolling_regression_out_sample(reg_lookback_days, r)
            r += 1
            
            ratio_ma = sma_diff_out_sample(lookback_days, y, slope)
            y += 1

            if pairs_trade == 0:

                if data['EPU'][i] > epu_threshold:

                    today_spread = np.log(data['Close Wheat'][i]) - slope * np.log(float(data['Close Corn '][i]))
                    if (today_spread - np.mean(ratio_ma))/ np.std(ratio_ma) > entry_zscore:

                        pairs_trade = 1
                        #Sell Corn and Buy Wheat
                        initial_corn_price = data['Close Corn '][i] * slope
                        initial_wheat_price = float(data['Close Wheat'][i])
                        trade_ratio = np.mean(ratio_ma)
                        trade_std = np.std(ratio_ma)
                        
                        slope_trade = slope
                        
                        
                        date_entered = i



                    elif (today_spread - np.mean(ratio_ma))/ np.std(ratio_ma) < -entry_zscore:

                        pairs_trade = -1
                        # Buy Corn and Sell Wheat 
                        initial_corn_price = data['Close Corn '][i] * slope
                        initial_wheat_price = float(data['Close Wheat'][i])
                        trade_ratio = np.mean(ratio_ma)
                        trade_std = np.std(ratio_ma)
                        
                        
                        slope_trade = slope

                        date_entered = i



            if pairs_trade == 1:

                today_spread = np.log(data['Close Wheat'][i]) - slope_trade * np.log(float(data['Close Corn '][i]))
                final_corn_price = data['Close Corn '][i] * slope_trade
                final_wheat_price = float(data['Close Wheat'][i]) 
                #PnL from trade 
                PnL = -((final_wheat_price - initial_wheat_price)/initial_wheat_price) + ((final_corn_price - initial_corn_price)/initial_corn_price)



                trade_pnl.append(PnL)
                today_date = i
                date_per_trade_pnl.append(today_date)

                initial_corn_price = data['Close Corn '][i] * slope_trade
                initial_wheat_price = float(data['Close Wheat'][i])
                if (today_spread - trade_ratio)/ trade_std < exit_zscore:


                    date_exited = i
                    #buy back Corn and Sell wheat 

                    pairs_trade = 0
                    trade_number += 1
                    days_in_trade_for_particular_trade = subtract_dates(date_entered, date_exited)
                    days_in_trade.append(days_in_trade_for_particular_trade)



                if PnL <= -20:

                    date_exited = i

                    pairs_trade = 0
                    
                    trade_number += 1
                    trade_stop_loss += 1
                    days_in_trade_for_particular_trade = subtract_dates(date_entered, date_exited)
                    days_in_trade.append(days_in_trade_for_particular_trade)




            elif pairs_trade == -1:

                today_spread = np.log(data['Close Wheat'][i]) - slope_trade * np.log(float(data['Close Corn '][i]))
                final_corn_price = data['Close Corn '][i] * slope_trade
                final_wheat_price = float(data['Close Wheat'][i]) 
                #PnL from trade 
                PnL = -((final_corn_price - initial_corn_price)/initial_corn_price) + \
                ((final_wheat_price - initial_wheat_price)/initial_wheat_price)

                trade_pnl.append(PnL)
                today_date = i
                date_per_trade_pnl.append(today_date)


                initial_corn_price = data['Close Corn '][i] * slope_trade
                initial_wheat_price = float(data['Close Wheat'][i])
                if (today_spread - trade_ratio)/ trade_std > -exit_zscore:

                    date_exited = i

                    pairs_trade = 0
                    trade_number += 1
                    days_in_trade_for_particular_trade = subtract_dates(date_entered, date_exited)
                    days_in_trade.append(days_in_trade_for_particular_trade)



                if PnL <= -20: 

                    date_exited = i

                    pairs_trade = 0
                    trade_number += 1
                    trade_stop_loss += 1
                    days_in_trade_for_particular_trade = subtract_dates(date_entered, date_exited)
                    days_in_trade.append(days_in_trade_for_particular_trade)




        #calculate sharpe 
        #average of PnL
        avg_pnl = np.mean(trade_pnl)
        #sharpe 
        sharpe_before_adj = avg_pnl/np.std(trade_pnl)
        sharpe = sharpe_before_adj * math.sqrt(252)
        sharpe = sharpe - (0.0119/(np.std(trade_pnl)*math.sqrt(252)))
        

    return sharpe, date_per_trade_pnl, trade_pnl, trade_number, trade_stop_loss, days_in_trade


sharpe_epu, date_epu, pnl_epu, trade_number_epu, trade_stop_loss_epu, days_in_trade_epu = calc_sharpe_corn_wheat_diff_out_sample(2.4, 2.2, 20, 360, 180)
print('Sharpe:', sharpe_epu)
print('Mean returns:', np.mean(pnl_epu))
print('Number of Trades:', trade_number_epu)
print('Times hit stop loss of 20:', trade_stop_loss_epu)
print('Average days in trade', np.mean(days_in_trade_epu))

## placing 0s into my daily pnl whenever I'm not in a trade 

cum_returns_pnl = []
counter = 0 
for i in data['Date']['2017-01-03':'2020-03-06']:
    if i in date_epu: 
        cum_returns_pnl.append(pnl_epu[counter].tolist()[0])
        counter += 1
    else: 
        cum_returns_pnl.append(0)


date_list_for_period = data['Date']['2017-01-03':'2020-03-06']


#now trying to graph cumulative returns 
        
    

df_cum_returns = {'Dates': date_list_for_period, 'returns': (np.add(1,cum_returns_pnl)).cumprod()}
df_final_returns = pd.DataFrame(df_cum_returns, columns=['Dates','returns'])

df_final_returns.plot(title="Cumulative Returns for (2.4, 2.2, 20, 360, 180) Out Sample \n \
(Entry, Exit, Commodity Lookback, EPU Lookback, RR Lookback)",x='Dates', y='returns')

plt.xticks(rotation = 45)
plt.show()


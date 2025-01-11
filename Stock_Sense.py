"""Author: Talha 
Computer Science Department @ Toronto Metropolitan University
11 January 2025 @ 2:30PM

-----------------Project_Overview----------------- :

---StockSense--- an advanced algorithmic trading system implemented in Python, designed to provide comprehensive market analysis and automated 
trading capabilities. The system leverages real-time market data integration through yfinance to analyze multiple assets including major stocks 
like S&P500, AAPL, TSLA, and MSFT.Combining traditional technical indicators such as Simple Moving Averages (50-day and 200-day) with complex
statistical analysis. At its core, the platform features robust data processing pipelines, integrated technical analysis tools, and a flexible 
backtesting framework that enables traders to evaluate and optimize their strategies across different market conditions.

The system's architecture encompasses multiple components including advanced visualization tools for generating correlation heatmaps and 
risk/return scatter plots, and an intelligent position sizing system based on volatility analysis. The platform implements both simple and 
logarithmic returns calculations,features drawdown analysis capabilities, and includes a multi-factor signal generation system for trading
decisions. Performance analytics are integrated throughout, providing key metrics such as maximum drawdown, and win rate calculations,which
makes it a complete solution for quantitative traders and analysts seeking to develop and test trading strategies in a systematic manner.

"""
import yfinance as yf 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 


#Initially I am taking any 4 random Stocks which are S&P500, APPLE, TESLA and Microsoft
ticker=["SPY","AAPL","TSLA","MSFT"]
stocks=yf.download(ticker,start="2016-01-01",end="2025-01-09")
# print(stocks)
print(stocks.head())
print(stocks.tail())
print(stocks.info())
print(stocks.to_csv("Stocks_data.csv"))
stocks=pd.read_csv("stocks_data.csv")
stocks=pd.read_csv("stocks_data.csv",header=[0,1],index_col=[0],parse_dates=[0])

#stocks.columns will give you name of all columns

#convert multiindex to one tuple 
stocks.columns=stocks.columns.to_flat_index()
print(stocks.columns)
print(stocks)
stocks.columns=pd.MultiIndex.from_tuples(stocks.columns) 
print(stocks)
print(stocks.columns)

stocks.describe=stocks.describe()
print(stocks.describe)

close=stocks.loc[:,"Close"].copy()
print(close)


import matplotlib.pyplot as plt 
plt.style.use("seaborn-v0_8")

normclose=close.div(close.iloc[0]).mul(100)
print(normclose.plot(figsize=(15,8),fontsize=12))
print(plt.legend(fontsize=12))
print(plt.show())



aapl=close.AAPL.copy().to_frame()

aapl['lag1']=aapl.shift(periods=1)
aapl.AAPL.sub(aapl.lag1)

aapl['Diff']= aapl.AAPL.sub(aapl.lag1)
aapl['% Change']=aapl.AAPL.div(aapl.lag1).sub(1).mul(100)

#if you want to rename any column :
aapl.rename(columns={"% Change":"Change"}, inplace=True)
print(aapl)
aapl1=aapl.AAPL.resample("ME").last() #use resample function to get monthly data instead of daily data.
print(aapl1)
aapl1=aapl.AAPL.resample("BME").last().pct_change(periods=1).mul(100) # if you want Last Business Day of the month Data with % change on montly frame
print(aapl1)

del aapl["Change"]
del aapl["lag1"]
del aapl["Diff"] # if you want to delte any column from your data.
print(aapl)


""" Let's Calculaate retrun on daily basis"""
ret=aapl.pct_change().dropna()
print(ret)

"""Let's Create a Histogram of any STOCK as we earlier did normal line graph"""
ret.plot(kind='hist',figsize=(12,8),bins=100)
print(plt.show())

"""Let's Calculate Mean return"""
daily_mean_return=ret.mean()
print(daily_mean_return)

"""Let's Calculate Variance"""
var_daily_=ret.var()
print(var_daily_)

"""Let's Calculate  Daily Standard_deviation"""
print(ret.std())

"""Calculating Annual Mean Return"""
annual_mean_return=daily_mean_return*252
print(annual_mean_return)

"""Let's Add more STOCKS of Your Choice"""
ticker=["SPY","AAPL","TSLA","MSFT","IBM","KO","DIS"] # Here I have added 3 more Stocks which are IBM , Coca cola and Disney
stocks=yf.download(ticker,start="2016-01-01",end="2025-01-09") # I have been fetching data of last 10 years and you can go further more if you want.

close=stocks.loc[:,"Close"].copy()
normclose=close.div(close.iloc[0]).mul(100)
print(normclose.plot(figsize=(15,8),fontsize=12))
print(plt.legend(fontsize=12))
print(plt.show()) # it will generate the Each Stock Prices throughout the decade 

ret=close.pct_change().dropna()
ret.head()
print(ret)
print(ret.describe().T) # this function will generate the information of each stock

summary=ret.describe().T.loc[:,["mean","std"]] # if you want to access two columns
print(summary) #here we are getting daily mean and standard deviation.

summary["mean"]=summary["mean"]*252
summary["std"]=summary["std"]*np.sqrt(252)
print(summary) # here we are getting anually mean and standard deviation.

"Let's take X-Axis as Standard Deviation return and Y-Axis as Mean return"
summary.plot.scatter(x="std",y="mean",figsize=(12,8),s=50,fontsize=15)


for i in summary.index:
    plt.annotate(i,xy=(summary.loc[i,"std"]+0.002,summary.loc[i,"mean"]+0.002),size=16)

plt.xlabel("Annual Risk(std)",fontsize=16)
plt.ylabel("Annual Return ",fontsize=16)
plt.title("Risk/return",fontsize=30)
print(plt.show())


""" Correlation and Covariance"""
print(ret)
ret.cov()
ret.corr() 

"""Import Seaborn Library in order to create a Heat MaP"""
import seaborn as sns
plt.figure(figsize=(12,8))
sns.set(font_scale=1.4)
sns.heatmap(ret.corr(),cmap="Reds",annot=True,annot_kws={"size":15},vmax=0.6) # as it plot rectangular data as acolor-encoded matrix
print(plt.show()) #it will generate correlation Graph and therefore it demonstrates us which stock we should invest in for long term.


"""Simple Returns and Log Returns"""
data_frame=pd.DataFrame(index=[2016,2017,2018],data=[100,50,95],columns=["Price"])
print(data_frame)

simple_returns=data_frame.pct_change().dropna()
print(simple_returns) #output will that a decline of 50% in 2017 and gain of 90% in 2018
print(simple_returns.mean()) # Meas returns are misleading as crazy thats why Log Returns comes into prespective.


log_returns=np.log(data_frame/data_frame.shift(1)).dropna()
print(log_returns)
print(log_returns.mean()) # Log Returns are Accurate 


"""Let's Calculate Moving Average
Below we are doing operation on S&P 500 """
SPY=yf.download("SPY")
spy=SPY[['Close']]
print(spy)

# We Generated Graph of S&P 500 (SPY) Graph and therefore from this we can calculate  Moving Average from this data.

"""Rolling Function: Performs Roling Window Calculations"""
spy_roll=spy.rolling(window=10)
print(spy_roll.mean())

spy["SMA50"]=spy.Close.rolling(window=50,min_periods=50).mean() # here we generated 50 Day Movig Average
print(spy) 
spy.plot(figsize=(12,8),fontsize=15)
print(plt.legend(loc="upper left", fontsize=15))
print(plt.show())

spy["SMA200"]=spy.Close.rolling(window=200,min_periods=200).mean() # here We generated 200 Day Moving Average
print(spy) 
spy.plot(figsize=(12,8),fontsize=15)
print(plt.legend(loc="upper left", fontsize=15))
print(plt.show())



"""
Cummulative Returns , Drawdowns etc..."""
apple=yf.download("AAPL")

apple["Daily_returns"]=np.log(apple["Close"]/apple["Close"].shift(1))
apple.dropna(inplace=True)
print(apple)


total_return=apple.Daily_returns.sum()
actual_return=np.exp(total_return) # if you had put 1$ in AAPL in 1980 
print(actual_return) # this amount of worth your 1$ would be today :$2396.43

apple["cummreturns"]=apple.Daily_returns.cumsum().apply(np.exp)
print(apple)

"""Plotting the Graph APPLE Stock Since 1980 till Today's Market"""
apple.cummreturns.plot(figsize=(12,8),title="APPLE Buy and Hold",fontsize=12)
print(plt.show())



""" How to Calculate Drawdowns.."""
apple["cumm_max"]=apple.cummreturns.cummax()
print(apple)

"""Now Plot the Graph"""
apple[["cummreturns","cumm_max"]].plot(figsize=(12,8),title="APPLE Buy and Hold+cumm_max",fontsize=12)
print(plt.show())

#Calculate Drawdowns (cumm_max - cummreturns) of APPLE 
apple["drawdown"]=apple["cumm_max"]-apple["cummreturns"]
print(apple)

#Calculate Percentage Drowdown: 
apple["drawdown %"]=(apple["cumm_max"]-apple["cummreturns"])/apple["cumm_max"]
print(apple)

#Calculate Maximum Drawdown of APPLE we ever have:
print(apple.drawdown.max())


# Maximum Drowdown Happened at ???
print(apple.drawdown.idxmax())


"""
Creating and Backtesting Strategy

SMA Strategy"""
data=apple[['Close']].loc[(apple.index>='1991-01-01')]
print(data)

#Now Let's calculate Moving Averages by using rolling function.
sma_s=50 # short period moving average
sma_l=100  # long Period moving average

data["sma_s"]=data.Close.rolling(sma_s).mean()
data["sma_l"]=data.Close.rolling(sma_l).mean()
data.dropna(inplace=True)
print(data)


# Plot the GRAPH
data.plot(figsize=(12,8),title="APPLE - SMA{} | SMA{}".format(sma_s,sma_l),fontsize=12)
print(plt.show())

# We can Test Apple Stock for a particular year: "2018"
data.loc["2018"].plot(figsize=(12,8),title="APPLE - SMA{} | SMA{}".format(sma_s,sma_l),fontsize=12)
print(plt.show())

"""Let's Discuss about when do we need a Position ?
When sma_s is greater than sma_l then we go long otherwise short (-1)"""
data["position"]=np.where(data["sma_s"]>data["sma_l"],1,-1)
print(data)


# Creating our First Function:
def test_strategy(stock,start,end,SMA):
    df=yf.download(stock,start=start,end=end)
    data=pd.DataFrame(df['Close'],columns=['Close'])
    data["returns"]=np.log(data['Close'].div(data['Close'].shift(1)))

    data["SMA_S"]=data['Close'].rolling(int(SMA[0])).mean()
    data["SMA_L"]=data['Close'].rolling(int(SMA[1])).mean()
    data.dropna(inplace=True)

    data["position"]=np.where(data["SMA_S"]>data["SMA_L"],1,-1)
    data["strategy"]=data["returns"]*data['position'].shift(1)
    data.dropna(inplace=True)
    ret=np.exp(data["strategy"].sum())
    std=data["strategy"].std()*np.sqrt(252)
    return ret, std 
test=test_strategy("SPY","2000-01-01","2025-01-01",(50,200))
print(test)
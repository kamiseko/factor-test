#!/Tsan/bin/python
# -*- coding: utf-8 -*-

# Libraries To Use
from __future__ import division
import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.stats
import os
from sklearn import linear_model
from datetime import datetime, time, date
import matplotlib.pyplot as plt
import seaborn as sns
import config as cf

data_path = cf.datapath
#timeStampNum = 500
#thresholdNum = 0.2


# get the list of ST, new  and delisted stocks on given date
# Return: LIST  that contains all badass stocks!
# Inputs:
# Date: TIMESTAMP or DATETIME , basically it's retrieved from the datelist
# stDF: DATAFRAME, contains the all stocks with ST FLAG
# tradeDayDF: DATAFRAME, which specifies the TRADING  DAY of all stocks
# stopFlagDF: DATAFRAME, which specifies the DELIST stocks
def GetSTNewSuspend(Date, stDF, tradeDayDF, stopFlagDF):  # Date is DateFrame time Index with hour second
    tempoDF1 = stDF.loc[Date]
    tempoDF2 = tradeDayDF.loc[Date]
    tempoDF3 = stopFlagDF.loc[Date]
    stockList1 = tempoDF1[~tempoDF1.isnull()].index.tolist()
    stockList2 = tempoDF2[tempoDF2 < 60].index.tolist()
    stockList3 = tempoDF3[~tempoDF3.isnull()].index.tolist()
    totalList = set(list(set(stockList1) | set(stockList2) | set(stockList3)))
    return list(totalList)


# This function to get he end of trading day of each month!
# Return: two LIST, that contains the start and end date of each month!
# Inputs:
# datetimeIndex : dataframe.index/series.index
def getLastDayOfMonth(datetimeIndex):     # This datetimeIndex should be chosen as dataframe.index
    timeTuple = sorted(list(set(zip(datetimeIndex.year,datetimeIndex.month))))
    #print timeTuple
    startOftheMonth = []
    endOftheMonth = []
    for time in timeTuple:
        targetList = datetimeIndex[(datetimeIndex.year==time[0])&(datetimeIndex.month==time[1])]
        startOftheMonth.append(targetList[0])
        endOftheMonth.append(targetList[-1])
    return startOftheMonth, endOftheMonth

# prepare data. filter data by the length of valid data
# Return: DATAFRAME
# Inputs:
# filename: STRING, the filename of the csv file
# timeStampNum: INT, the time period of the backtest
# ThresholdNum: INT, the thresholdNum of valid data to drop na
def getData(filename, timeStampNum, thresholdNum):
    factorData = pd.read_csv(data_path+filename, infer_datetime_format=True, parse_dates=[0], index_col=0)
    factorData = factorData[-timeStampNum-1:-5]
    enoughDataStock = factorData.isnull().sum() < (timeStampNum*thresholdNum)
    enoughDataStockList = enoughDataStock[enoughDataStock == True].index.tolist()
    factorData = factorData[enoughDataStockList]
    factorData = factorData.fillna(method='ffill')   ### Fill N/A value with the last valid obsevation
    factorData = factorData.fillna(method='bfill')
    return factorData

# ------------------------------- the following part is to winsorize normalize and neutralize-------------------------
# Winsorize and  Normalize in One function!!
# The following function is designed to Winsorize by  MAD METHOD! U can also define the method by ur self. \
# Just like the adjustedBoxplot method below.
# Return a DATAFRAME
# Inputs :
# data: DATAFRAME that has NO NAN VALUE
# filterdict: DICTIONARY .the filtered stocks that  obtained from GetSTNewSuspend ,\
# KEY is the date and VALUE is the stocks that should
# filtered out  on that date.
# datelist : LIST. that contains the date.
def winsorAndnorm(data, filterdict, datelist):
    dataWinsorized = data.copy()
    for date in datelist:
        remainedStocks = list(set(data.columns.tolist()) - set(filterdict[date]))
        filteredStocks = list(set(data.columns.tolist()) - set(remainedStocks))
        dataWinsorized.loc[date][filteredStocks] = np.NaN
    dataWinsorizedTrans = dataWinsorized.loc[datelist].T
    MAD = 1.483*np.abs(dataWinsorizedTrans-dataWinsorizedTrans.median(skipna=True))
    dataWinsorizedTrans[dataWinsorizedTrans > dataWinsorizedTrans.median(skipna=True)+3*MAD] = \
        dataWinsorizedTrans.median(skipna=True)+3*MAD
    dataWinsorizedTrans[dataWinsorizedTrans < dataWinsorizedTrans.median(skipna=True)-3*MAD] = \
        dataWinsorizedTrans.median(skipna=True)-3*MAD
    return ((dataWinsorizedTrans - dataWinsorizedTrans.mean(axis=0, skipna=True))/dataWinsorizedTrans.std(axis=0, skipna=True)).T


# Boxplot method adjusted by MedCouple to winsorize
# Return two FLOAT
# Input : ARRAY
def adjustedBoxplot(a): # x is a np.array
    medianofa = np.median(a)
    lowergroup = a[a <= np.median(a)]
    uppergroup = a[a >= np.median(a)]
    q75, q25 = np.percentile(a, nparray([75, 25]))
    iqr = q75 - q25
    resultlist = []
    for i in lowergroup:
        for j in uppergroup:
            if j > i:
                resultlist.append((j+i-2*medianofa/j-i))
            else:
                resultlist.append(np.sign(len(uppergroup)-j-i-1))
    p = np.median(resultlist)
    L = q25 - 1.5*np.exp(-3.5*p)*iqr if p >= 0 else q25-1.5*np.exp(-4*p)*iqr
    U = q75 + 1.5*np.exp(4*p)*iqr if p >= 0 else q75+1.5*np.exp(3.5*p)*iqr
    return L, U

# Neutralize factor
# Returns a DATAFRAME
# Inputs are like:
# normalizedFactorDF: DATAFRAME , the FACTOR data that was  winsorized and normalized.
# normalizedLFCAPDF:  DATAFRAME , the CIRCULATION MARKET VALUE that was winsorized and normalized and Take log
# IndustryDF : DATAFRAME , the Industry Class u use, default it's ZX INDUSTRY
# datelist : LIST , date list should be same through all functions!
def neutralizeFactor(normalizedFactorDF, normalizedLFCAPDF, IndustryDF, datelist):
    factorNeutralized = pd.DataFrame(index=normalizedFactorDF.index, columns=normalizedFactorDF.columns, data=None)
    for date in datelist:
        LFCAPIndice = normalizedLFCAPDF.loc[date].dropna()
        factorIndice = normalizedFactorDF.loc[date].dropna()
        intersectionStocks = list(set(LFCAPIndice.index) & set(factorIndice.index))
        dummy_Matrix = pd.get_dummies(IndustryDF.loc[date]).T.iloc[:-1]
        dummy_Matrix = dummy_Matrix[intersectionStocks].append(LFCAPIndice.loc[intersectionStocks])
        result = sm. OLS(factorIndice.loc[intersectionStocks].T, dummy_Matrix.T).fit()
        factorNeutralized.loc[date][intersectionStocks] = result.resid
    return factorNeutralized

### This function is to generate industry dummy matrix
def  generateIndDF(data_path,filename,timeStamp):
    InData = pd.read_csv(data_path+filename, infer_datetime_format=True,parse_dates=[0], index_col=0)
    #InData= InData.tail(timeStamp+5)[-timeStamp-1:-1].dropna(axis=1,how='any')
    InData = InData.tail(timeStamp+5)[-11:-1].dropna(axis=1, how='any')
    InduNum = int(InData.max().max()-InData.min().min())
    x = range(InduNum)
    DummyDF = pd.DataFrame(index=x, columns=InData.columns.tolist(), data=None)
    for stk in DummyDF.columns.tolist():
        Tag = int(InData[stk].iloc[-1])
        if Tag == InduNum+1:
            pass
        else:
            DummyArray = np.zeros(InduNum)
            DummyArray[Tag-1] = 1
            DummyDF[stk] = DummyArray
    return DummyDF

# -------------------------the following part is to calculate the monthly return/IC of the given factor----------------

# To calculate the return or the active return of given date
# return: DATAFRAME ,that contains the RETURN/ACTIVE RETURN of each stock
# Inputs:
# priceData: DATAFRAME ,which is obtained from getData function
# benchmark: DATAFRAME, same as priceData but the filter by threshold part could be deleted
# datelist: LIST ,which contains the dates.
# activeReturn : Boolean Value. True to calculate ACTIVE RETURN
def calcReturn(priceData,datelist,benchmark = None,activeReturn = True, logReturn = True):
    returnOfStocks = np.log((priceData.loc[datelist].shift(-1)/priceData.loc[datelist]).iloc[:-1]) if logReturn is True\
            else priceData.loc[datelist].pct_change().shift(-1).iloc[:-1]
    if activeReturn:
        returnOfBenchmark = np.log((benchmark.loc[datelist].shift(-1)/benchmark.loc[datelist]).iloc[:-1]) if logReturn \
            is True else benchmark.loc[datelist].pct_change().shift(-1).iloc[:-1]
        activeReturn = returnOfStocks.apply(lambda x: x - returnOfBenchmark)
        return activeReturn
    return returnOfStocks


# This is to calculate monthly return fo the given factor, P-value, T-value, and  IC.
# Update the four Given DATAFRAME
# Inputs:
# factorNeutralized : DATAFRAME ,the dataframe get from neutralizeFactor()
# activeReturn : DATAFRAME , get from calcReturn()
# facotName : STRING , the name of the factor.
# Warning : THE FOUR DATAFRAME SHOULD DEFINED BEFORE USING THIS FUNCTION
def calReturnAndIC(returnofFactor,tValueofFactor,pValueofFactor,ICFactor,factorNeutralized,activeReturn,factorName):
    for date in activeReturn.index:
        factorIndice = factorNeutralized.loc[date].dropna()
        activeReturnIndice = activeReturn.loc[date].dropna()
        intersections = list(set(factorIndice.index) & set(activeReturnIndice.index))
        result = sm.OLS(activeReturnIndice.loc[intersections], factorIndice.loc[intersections].astype(float)).fit()
        returnofFactor.loc[date][factorName] = result.params[0]
        tValueofFactor.loc[date][factorName] = result.tvalues[0]
        pValueofFactor.loc[date][factorName] = result.pvalues[0]
        ICFactor.loc[date][factorName] = activeReturnIndice.loc[intersections].corr(factorIndice.loc[intersections].astype(float),\
                                                                                   method='spearman')
    #return returnofFactor, tValueofFactor, pValueofFactor, ICFactor


# This function is to calculate the stocks grouped based by factor value,u cam also use 'quantile' method to get the \
# groups
def getStockGroup(factorData, groupNum=10, Mean_Num=20, ascendingFlag = True):
    #factorData is disposed data which have been winsorized, normalized and neutralized
    groupDic = {}
    if Mean_Num == 1:
        sortedStk = factorData.iloc[-1].dropna().sort_values(ascending=ascendingFlag)
    else:
        sortedStk = factorData.iloc[-Mean_Num:].mean(skipna=True).sort_values(ascending=ascendingFlag)
    #print factorData.index[-1]
    stkNumPerGFloor = int(np.floor(len(sortedStk)/groupNum))
    stkNumPerGCeil = int(np.ceil(len(sortedStk)/groupNum))
    remainderCount = int(np.mod(len(sortedStk),groupNum))
    for i in xrange(groupNum):
        if i < remainderCount:
            groupDic['group'+'_'+str(i)] = sortedStk[i*stkNumPerGCeil:(i+1)*stkNumPerGCeil].index.tolist()
        elif i == remainderCount:
            groupDic['group'+'_'+str(i)] = sortedStk[i*stkNumPerGCeil:i*stkNumPerGCeil+stkNumPerGFloor].index.tolist()
        else:
            groupDic['group'+'_'+str(i)] = sortedStk[remainderCount+\
                                                     i*stkNumPerGFloor:remainderCount+(i+1)*stkNumPerGFloor].index.tolist()
    return groupDic

# this is to show the corelation between two risk factors
# Return: Dataframe that contains both PEARSON and SPEARMAN correlation
# Input:
# factor1: DATAFRAME, DF of factor1(can either be the raw data or the nuetralized one)
# factor2: DATAFRAME
# datelist: LIST, which contains the date u want to calc correlation
# filterdic: DICTIONARY, the KEY of which is the Date of datelist and the VALUE is LIST of the filtered stocks
# \Same as winsorAndnorm function
def showCorrelation(factor1, factor2, datelist, filterdic = None):
    corrDF = pd.DataFrame(index=datelist, columns=['Pearson', 'Spearman'], data=None, dtype=float)
    for date in datelist:
        factorIndice1 = factor1.loc[date].dropna()
        factorIndice2 = factor2.loc[date].dropna()
        if not filterdic:
            intersections = list(set(factorIndice1.index) & set(factorIndice2.index))
        else:
            intersections = list((set(factorIndice1.index) & set(factorIndice2.index)) - set(filterdic[date]))
        factorTrueValue1 = factorIndice1.loc[intersections].astype(float)
        factorTrueValue2 = factorIndice2.loc[intersections] .astype(float)
        corrDF.loc[date]['Pearson'] = factorTrueValue1.corr(factorTrueValue2.loc[intersections],
                                                                                   method='pearson')
        corrDF.loc[date]['Spearman'] = factorTrueValue1.corr(factorTrueValue2.loc[intersections],
                                                                                   method='spearman')
    return corrDF


# this is to update the  forward adjusted Price
# this calculation need  the additional files, namely AdjustedFactor and Price dataframe
def calAdjustedPrice():
    # Adjusted factor
    AdjFacBackward=pd.read_csv(data_path+filenameAdjustFactor,infer_datetime_format=True,parse_dates=[0],index_col=0)
    AdjFacBackward=AdjFacBackward[:]

    #PriceData to Adjust
    PriceToAdj=pd.read_csv(data_path+filenamePirce,infer_datetime_format=True,parse_dates=[0],index_col=0)
    PriceToAdj=PriceToAdj[:]

    #Calculate
    AdjFacforward = AdjFacBackward/AdjFacBackward.max()
    adjustedPrice = (AdjFacforward*PriceToAdj).round(5)
    adjustedPrice.to_csv(data_path+'my_own_factor_AdjustedPriceForward.csv')





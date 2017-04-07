#!/Tsan/bin/python
# -*- coding: utf-8 -*-

# Libraries to use
from __future__ import division
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import statsmodels
import cvxopt as cv
from cvxopt import solvers

# Import My own library for factor testing
import factorFilterFunctions as ff


#  own method to cal moving  weighted Covariance Matrix
def calEWMcovariance(facRetdf, decay=0.94):
    '''
    To calculate EWM covariance matrix of given facRetdf
    output: Dataframe, the ewm cov-matrix of the factors
    input:
    facRetdf: Dataframe, factor return dataframe
    decay: float, decay-factors
    Decay factors were set at:
    − 0.94 (1-day) from 112 days of data;
    − 0.97 (1-month) from 227 days of data.
    '''
    m, n = facRetdf.shape
    facDF = facRetdf - facRetdf.mean()
    for i in xrange(m):
        facDF.iloc[i] = np.sqrt(decay**(m-1-i)) * facDF.iloc[i]
    ewmCovMatrix = facDF.T.dot(facDF) * (1-decay)/(1-decay**m)
    return ewmCovMatrix

# get intersection
def getInterStk(dfList):
    '''
    To get the columns intersections of several dataframe.
    output: List, the intersection of the columns of dataframes.
    input:
    dfList: List, which contains dataframe u want to get the intersection, len(dfList) should be more than 1.
    '''
    columnsList = map(lambda x: set(x.columns.tolist()), dfList)
    stkList = reduce(lambda x, y: x & y, columnsList)
    return stkList

# independentfactor should be a list contains of dataframe
def orthoFactor(factordf, independentfactor, WLS =False, weightdf = None):
    '''
    Muti variable regression for return.
    returndf and dataframes in factorDict should have same index and same columns
    output: Dataframe, the orthogonalized result of factordf
    input:
    factordf: Dataframe, factor to be orthogonalized
    independentfactor: List,  the values are the factor dataframe as independence in regression(all \
    with same columns and index)
    WLS: True to use WLS , False to use OLS. If True, then weightdf should not be none.
    weightdf: Dataframe , which has no nan and the shape is same as dataframes in factorDict.
    '''
    emptydf = pd.DataFrame(index=factordf.index, columns=factordf.columns, data=None, dtype=float)
    dfNum = len(independentfactor)
    if dfNum == 0:
        print 'Input is an empty list!'
        raise ValueError
    for date in factordf.index:
        factordfSlice = factordf.loc[date]
        mapfunction = map(lambda x: x.loc[date], independentfactor)
        if dfNum > 1:
            totaldf = pd.concat(mapfunction, axis=1)
        else:
            totaldf = independentfactor[0].loc[date]
        if WLS:
            w = weightdf.loc[date]
            result = sm.WLS(factordfSlice.T, totaldf, weights=1/w).fit()
        else:
            result = sm.OLS(factordfSlice.T, totaldf).fit()
        emptydf .loc[date] = result.resid
    return emptydf

# construct the multiple factor structural risk model
def multiFactorReg(returndf,factorDict,WLS =False, weightdf = None):
    '''
    Multi variable regression for return.
    returndf and dataframes in factorDict should have same index and same columns.
    output: 4 Dataframe, respectively idiosyncratic return for each stock, factor Return, factor P-value and
    R-Square of the linear regression model.
    input:
    returndf: Dataframe, can either be return or acticve return.
    factorDict: Dictionary, the keys are the names of factors and the values are the corresponding factor dataframe(all\
    with same columns and index).
    WLS: True to use WLS , False to use OLS. If True, then weightdf should not be none.
    weightdf: Dataframe , which has no nan and the shape is same as dataframes in factorDict.
    '''
    specificReturn = pd.DataFrame(index=returndf.index, columns=returndf.columns, data=None, dtype=float)
    factorReturn = pd.DataFrame(index=returndf.index, columns=factorDict.keys(), data=None, dtype=float)
    factorPvalue = pd.DataFrame(index=returndf.index, columns=factorDict.keys(), data=None, dtype=float)
    RSquare = pd.DataFrame(index=returndf.index, columns=['R-Square'], data=None, dtype=float)
    dfNum = len(factorDict.keys())
    if dfNum == 0:
        print 'Input is an empty list!'
        raise ValueError
    for date in returndf.index:
        returndfSlice = returndf.loc[date]
        mapfunction = map(lambda x: x.loc[date], factorDict.values())
        if dfNum > 1:
            totaldf = pd.concat(mapfunction, axis=1)
        else:
            totaldf = factorDict.values()[0].loc[date]
        if WLS:
            w = weightdf.loc[date]
            result = sm.WLS(returndfSlice.T, totaldf, weights=1/w).fit()
        else:
            result = sm.OLS(returndfSlice.T, totaldf).fit()
        specificReturn .loc[date] = result.resid
        factorReturn .loc[date] = result.params.values
        factorPvalue . loc[date] = result.pvalues.values
        RSquare .loc[date] = result.rsquared
    return specificReturn, factorReturn, factorPvalue, RSquare
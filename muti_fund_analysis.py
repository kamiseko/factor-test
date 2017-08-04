#!/Tsan/bin/python
# -*- coding: utf-8 -*-

# Libraries To Use
from __future__ import division
import numpy as np
import pandas as pd
import statsmodels.api as sm
import os
from datetime import datetime,time,date
import matplotlib.pyplot as plt
import seaborn as sns
import mysql.connector
import cvxopt as cv
from cvxopt import solvers
from sklearn.covariance import ShrunkCovariance
import scipy as sp
import json
from sqlalchemy import create_engine
import sqlalchemy.engine.url as url


tableName ='fund_nv_standard_w' # table to query
indexID = 'hs300' # benchmark



# 数据库连接cursor
with open('conf.json', 'r') as fd:
    conf = json.load(fd)
cnx = mysql.connector.connect(**conf['src_db'])

# 获取数据的函数
def get_simplied_fund_nw(fundID, tableName=tableName):
    '''Get simplied fund data which has only two columns: date and the corresponding networth
    Input:
    fundID: String, the fundID of the fund, JRXXXXX alike.
    tableName : String, the corresponding table name of the fund net worth data
    Return: DataFrame, which has only two columns: date and the corresponding networth,
    and names of the columns are respectively date and fundID.
    '''

    try:
        # sql_query='select id,name from student where  age > %s'
        cursor = cnx.cursor()
        sql = "select statistic_date,swanav from %s where fund_id = '%s'" % (tableName, fundID)
        cursor.execute(sql)
        result = cursor.fetchall()
    finally:
        pass
    pdResult = pd.DataFrame(result, dtype=float)
    pdResult.columns = ['date', fundID]
    pdResult = pdResult.drop_duplicates().set_index('date')
    pdResult = pdResult.dropna(axis=0)
    pdResult.index = pd.to_datetime(pdResult.index)
    pdResult = pdResult.fillna(method='ffill')
    # pdResult ['weekly_return'] = pdResult['net_worth'].pct_change()
    return pdResult

# to calShrunkCov
def calShrunkCov(weekly_return):
    '''calculate Shrunk Covariance. ie. a 10 asset return array should return 10 * 10 dimension array
    Input: N * K np.array .
    Output: N * N array'''
    return ShrunkCovariance(0.1).fit(weekly_return.dropna().values).covariance_

# 获取mean-std方法的权重
def cal_minRisk_weight(cov):
    '''To calculate weight of portfolio with min Risk
    Input: Cov: n-d array, obtained from df.values
    Output: 1-d array, the weight vector'''
    stkNum = cov.shape[1]
    P = cv.matrix(cov)
    q = cv.matrix(0.0, (stkNum, 1))
    G = cv.matrix(np.concatenate((np.diag(np.ones(stkNum)), - np.diag(np.ones(stkNum)))))
    h = cv.matrix(np.append(0.2 * np.ones(stkNum), np.zeros(stkNum)))
    A = cv.matrix(np.ones(stkNum)).T
    b = cv.matrix(1.0).T
    sol = solvers.qp(P, q, G, h, A, b)
    return  np.array(list(sol['x']))

# 风险评价的目标优化函数
def object_fun(x,return_cov):
    '''objective function for Risk budget portfolio
    Input: x ,n-d array ,the weight vector
           return_cov, the covariance of the asset return
    Output: ovjective function'''
    N = x.shape[0]   # get the number of asset
    covMatrix = np.matrix(return_cov)
    x = np.matrix(x)
    y = np.array(x) * (np.array(covMatrix * x.T).T)
    totalRisk = x * covMatrix * x.T
    b = totalRisk / N
    y = np.squeeze(np.asarray(y))  # return 1 dimension array in order to perform indexing
    totalY = y.sum()
    fval = 0
    for i in range(0,N):
        xi = (y[i]/totalY - b ) ** 2
        fval =fval +xi
    return fval


# 风险平价算法
def cal_RB_weight(cov):
    '''
    To calculate weight of portfolio with risk parity(the special case for risk budget portfolio)
    Input: Cov: n-d array, obtained from df.values
    Output: 1-d array, the weight vector
    '''

    bnds = ((0, 1),) * cov.shape[0]  # bounds for weights (number of bounds  = to number of assets)
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0})
    x0 = np.ones(cov.shape[0]) * 1 / cov.shape[0]
    res = sp.optimize.minimize(object_fun, x0, args=cov, method='SLSQP', constraints=cons, bounds=bnds)
    if res.success:
        allocation = res.x
        allocation[allocation < 0] = 0  # since fund can not be shorted!
        denom = np.sum(allocation)
        if denom != 0:  # normalization process
            allocation = allocation / denom
    else:
        allocation = x0
    return allocation


if __name__ == '__main__':
    fund_list = ['JR000007', 'JR000050', 'JR000052', 'JR000053', 'JR000169', 'JR000170', 'JR000171', 'JR000185',
                 'JR000186', 'JR000278']  # 备选基金
    NumOfAsset = len(fund_list)
    fund_data = map(get_simplied_fund_nw, fund_list)
    fund_data_merged = reduce(lambda x, y: x.merge(y, how='inner', left_index=True, right_index=True), fund_data)
    weekly_return = fund_data_merged.pct_change()
    # coefficient
    coeff_matrix = weekly_return.corr(method='pearson').round(4)
    coeff_matrix  # 基金收益相关性矩阵
    # visualization
    plt.figure(figsize=(16, 9))
    ax = plt.axes()
    sns.heatmap(coeff_matrix, ax=ax, annot=True)
    ax.set_title('Correlationship Matrix', fontsize=18, fontweight='bold')

    fund_cov = weekly_return.cov()

    # descriptive information
    descri_info = weekly_return.describe().round(4)  # 基金描述数据
    print descri_info

    # ------------------------------ minimize risk ---------------------------------
    tANetWorth = fund_data_merged / fund_data_merged.iloc[0]

    tACovMatrix = tANetWorth.ewm(ignore_na=True, min_periods=0, halflife=60).cov(pairwise=True).iloc[-1]

    endOfMonthList = tANetWorth.index
    positionSheetMR = pd.DataFrame(index=tANetWorth.index, columns=tANetWorth.columns, data=None, dtype=float)
    positionSheetRB = positionSheetMR.copy()

    # backtest (change position at the end of each month)
    returnDF = tANetWorth.pct_change()
    for date in tANetWorth.index:
        tempdf = returnDF.loc[:date]
        tempNWdf = tANetWorth.loc[:date]
        if date in endOfMonthList and tempdf.shape[0] >= 5 * NumOfAsset:  # burn in period

            temCovMatrix = calShrunkCov(tempdf.iloc[-5 * NumOfAsset:])
            # print temCovMatrix
            # Optimize
            '''stkNum = temCovMatrix.shape[1]
            P = cv.matrix(temCovMatrix)
            q = cv.matrix(0.0, (stkNum, 1))
            G = cv.matrix(np.concatenate((np.diag(np.ones(stkNum)), - np.diag(np.ones(stkNum)))))
            h = cv.matrix(np.append(0.2 * np.ones(stkNum), np.zeros(stkNum)))
            A = cv.matrix(np.ones(stkNum)).T
            b = cv.matrix(1.0).T
            sol = solvers.qp(P, q, G, h, A, b)'''

            positionSheetMR.loc[date] = cal_minRisk_weight(temCovMatrix)
            positionSheetRB.loc[date] = cal_RB_weight(temCovMatrix)

    positionSheet1 = positionSheetMR.shift(1).fillna(method='ffill').fillna(0)
    positionSheet2 = positionSheetRB.shift(1).fillna(method='ffill').fillna(0)

    mrpos = positionSheet1.iloc[-1]
    mrpos.name = 'Minimized_Risk'
    rppos = positionSheet2.iloc[-1]
    rppos.name = 'Risk_Parity'

    posref = pd.concat([mrpos, rppos], axis=1)
    posref['Equal_Weighetd'] = 1. / posref.shape[0]
    posref = posref.round(4)  # 最后一天的仓位数据

    # calculate networth and show the plot
    retMR = (positionSheet1 * weekly_return).sum(axis=1)
    retRB = (positionSheet2 * weekly_return).sum(axis=1)
    retEW = (weekly_return.sum(axis=1) / NumOfAsset)
    retEW.name = 'Equal_Weighetd'

    retMR.name = 'Minimized_Risk'
    retRB.name = 'Risk_Parity'

    retTotal = pd.concat([retMR, retRB, retEW], axis=1)
    retTotal.index.name = 'time'
    retTotal = retTotal.dropna()
    retTotal = retTotal.loc[retTotal[retTotal[retMR.name] == 0].index[-1]:]
    retTotal.iloc[0] = 0
    # retTotal = retTotal / retTotal.iloc[0]
    fig = plt.figure(figsize=(14, 9))
    # Add a subplot
    ax = fig.add_subplot(111)
    retTotal.cumsum().plot(figsize=(22, 14), ax=ax, title='Cumulative Return', fontsize=13)
    ax.set_title(ax.get_title(), alpha=0.7, fontsize=30, ha='right')

    retTotal = retTotal.round(4)  # 几种不同方法的净值曲线
    plt.show()

    db_engine = create_engine(
        'mysql://{0}:{1}@{2}:{3}/{4}'.format('tai', 'tai2015', '119.254.153.20', 13311, 'PrivateEquityFund_W',
                                             encoding='utf-8'))

    # coeff_matrix.reset_index().to_sql(name='multiple_funds_coeff',con=db_engine, if_exists='replace',index=False)
    # descri_info.T.reset_index().to_sql(name='multiple_funds_description',con=db_engine, if_exists='replace',index=False)
    # retTotal.reset_index().to_sql(name='multiple_funds_comb_ret',con=db_engine, if_exists='replace',index=False)
    # posref.reset_index().to_sql(name='multiple_funds_position', con=db_engine, if_exists='replace', index=False)

    cnx.close()






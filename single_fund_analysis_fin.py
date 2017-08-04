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

import mysql.connector
import json

# 数据库
with open('conf.json', 'r') as fd:
    conf = json.load(fd)
cnx = mysql.connector.connect(**conf['src_db'])

# 一些常量
riskFreeRate = 0.02
varThreshold = 0.05

hs300 = 'hs300'  # 沪深300
zz500 = 'csi500'  # 中证500

# 相关信息
fundID = 'JR000001'  # 周度更新
tableName ='fund_nv_standard_w'  # table to query

fund_info = 'fund_info'  # 基金信息表格

type_index_table = 'index_stype_code_mapping'  # 表格名称-不同基金种类对应的指数

fund_type_table = 'fund_type_mapping'  # 表格名称 - 基金种类表

index_table = 'fund_weekly_index'  # 表格名称 - 指数的表现


# 历史表现表头
header_historical_perform = ['total_return','return_m1','return_m3','return_m6','return_y1','return_this_year']

# 指标表头
columns_names =['net_worth','annualized_ret','annualized_vol',\
                'annualized_downside_risk','odds','max_dd','max_dd_start_date','max_dd_end_date',\
                 'sharpe_ratio','calmar_ratio','sortino_ratio','information_ratio_300','information_ratio_500','VaR','Pnl_ratio']


# 情景分析的必要参数
header_stress_test = ['situation','start_date','end_date','cumulative_ret','max_dd']
start1 =  datetime.strptime('20141101', '%Y%m%d').date()
end1 = datetime.strptime('20141231', '%Y%m%d').date()
start2 = datetime.strptime('20150601', '%Y%m%d').date()
end2 = datetime.strptime('20150831', '%Y%m%d').date()
start3 = datetime.strptime('20160101', '%Y%m%d').date()
end3 = datetime.strptime('20160131', '%Y%m%d').date()

# 函数定义

# GET THE  WEEKLY NET WORTH DATA OF EACH FUND
def get_fund_data(fundID,tableName =tableName):
    try:
        #sql_query='select id,name from student where  age > %s'
        cursor = cnx.cursor()
        sql = "select fund_id,statistic_date_std,swanav from %s where fund_id = '%s'" % (tableName,fundID)
        cursor.execute(sql)
        result = cursor.fetchall()
    finally:
        pass
    pdResult = pd.DataFrame(result,dtype =float)
    pdResult.columns = ['fund_id','date','net_worth']
    pdResult = pdResult.drop_duplicates().set_index('date')
    #pdResult = pdResult.set_index('date')
    pdResult = pdResult.dropna(axis=0)
    pdResult = pdResult.fillna(method = 'ffill')
    return pdResult

# 获取该基金更新频率
def get_data_frequency(fundID,tableName =fund_info):
    try:
        #sql_query='select id,name from student where  age > %s'
        cursor = cnx.cursor()
        sql = "select fund_id,data_freq from %s where fund_id = '%s'" % (tableName,fundID)
        cursor.execute(sql)
        result = cursor.fetchall()
    finally:
        pass
    return result

# 获取指数
def get_benchmark(indexID,tableName = 'market_index'):
    try:
        cursor = cnx.cursor()
        sql = "select %s,statistic_date from %s " % (indexID,tableName)
        cursor.execute(sql)
        result = cursor.fetchall()
    finally:
        pass
    pdResult = pd.DataFrame(result,dtype =float)
    pdResult.columns = [''+indexID+'','date']
    pdResult = pdResult.dropna(axis=0)
    pdResult = pdResult.drop_duplicates().set_index('date')
    pdResult = pdResult.fillna(method = 'ffill')
    return pdResult


#  获取该基金的分类代码
def get_fund_type(fundID, tableName=fund_type_table):
    try:
        # sql_query='select id,name from student where  age > %s'
        cursor = cnx.cursor()
        sql = "select fund_id,stype_code from %s where fund_id = '%s'" % (tableName, fundID)
        cursor.execute(sql)
        result = cursor.fetchall()
    finally:
        pass
    # pdResult = dict(result)
    pdResult = pd.DataFrame(result)
    pdResult.columns = ['' + fundID + '', 'stype_code']
    pdResult.set_index(fundID, inplace=True)
    pdResult = pdResult.dropna(axis=0)
    pdResult['filter'] = pdResult['stype_code'].apply(lambda x: 1 if str(x).startswith('601') else 0)  # 只有601开头的才有数据！
    pdResult = pdResult[pdResult['filter'] == 1]
    return pdResult

#  获取该基金的分类名称
def get_fund_type_name(typename,tableName = fund_type_table):
    try:
        #sql_query='select id,name from student where  age > %s'
        cursor = cnx.cursor()
        sql = "select fund_id,type_name from %s where type_name = '%s'" % (tableName,typename)
        cursor.execute(sql)
        result = cursor.fetchall()
    finally:
        pass
    #pdResult = dict(result)
    pdResult = pd.DataFrame(result)
    pdResult.columns = ['fund_id','type_name']
    pdResult.set_index('fund_id',inplace=True)
    pdResult = pdResult.dropna(axis=0)
    #pdResult['filter'] = pdResult['stype_code'].apply(lambda x : 1 if str(x).startswith('601') else 0) # 只有601开头的才有数据！
    #pdResult = pdResult[pdResult['filter'] ==1]
    return pdResult

# 私募指数基金分类表格对应（只需要跑一次）
def get_type_index_table(tableName = type_index_table):
    try:
        #sql_query='select id,name from student where  age > %s'
        cursor = cnx.cursor()
        sql = "select * from %s" % (tableName)
        cursor.execute(sql)
        result = cursor.fetchall()
    finally:
        pass
    #pdResult = dict(result)
    pdResult = pd.DataFrame(result)
    pdResult = pdResult.dropna(axis=0)
    pdResult.columns = [i[0] for i in cursor.description]
    pdResult.set_index('stype_code',inplace=True)
    return pdResult


# 私募指数净值的时间序列
def get_index(index, tableName=index_table):
    try:
        # sql_query='select id,name from student where  age > %s'
        cursor = cnx.cursor()
        sql = "select index_id,statistic_date,index_value from %s where index_id = '%s'" % (tableName, index)
        cursor.execute(sql)
        result = cursor.fetchall()
    finally:
        pass
    pdResult = pd.DataFrame(result, dtype=float)
    pdResult.columns = ['index', 'date', 'net_worth']
    pdResult = pdResult.drop_duplicates().set_index('date')
    pdResult = pdResult.dropna(axis=0)
    pdResult = pdResult.fillna(method='ffill')
    return pdResult

# 计算最大回撤 返回到一个dataframe,输入值为networth的series
def cal_max_dd_df(networthSeries):
    maxdd = pd.DataFrame(index = networthSeries.index, data=None, columns =['max_dd','max_dd_start_date','max_dd_end_date'],dtype = float)
    maxdd.iloc[0] = [0,maxdd.index[0],maxdd.index[0]]
    maxdd.is_copy = False
    for date in networthSeries.index[1:]:
        maxdd.loc[date] = [1 - networthSeries.loc[date] / networthSeries.loc[:date].max(),networthSeries.loc[:date].idxmax(),date]
        #maxdd[['max_dd_start_date','max_dd_end_date']].loc[date] = [[networthSeries.loc[:date].idxmax(),date]]
        #maxdd['max_dd_start_date'].loc[date] = networthSeries.loc[:date].idxmax()
    return maxdd

# 计算最大回撤返回三个指标
def cal_max_dd_indicator(networthSeries):
    maxdd = pd.DataFrame(index = networthSeries.index, data=None, columns =['max_dd','max_dd_start_date','max_dd_end_date'],dtype = float)
    maxdd.iloc[0] = 0
    maxdd.is_copy = False
    for date in networthSeries.index[1:]:
        maxdd.loc[date] = [1 - networthSeries.loc[date] / networthSeries.loc[:date].max(),networthSeries.loc[:date].idxmax(),date]
        #maxdd[['max_dd_start_date','max_dd_end_date']].loc[date] = [[networthSeries.loc[:date].idxmax(),date]]
        #maxdd['max_dd_start_date'].loc[date] = networthSeries.loc[:date].idxmax()
    return maxdd['max_dd'].max(), maxdd.loc[maxdd['max_dd'].idxmax]['max_dd_start_date'],maxdd.loc[maxdd['max_dd'].idxmax]['max_dd_end_date']

# 计算下行风险，返回一个series。输入为一个return的series
def cal_downside_risk(returnSeries):
    rs = returnSeries.copy()
    rs[rs > rs.mean()] =0
    return rs.std(skipna = True) * np.sqrt(ScaleParameter)

# 计算VAR 返回一个值
def cal_var(returnSeries ,alpha = varThreshold):
    return returnSeries.quantile(alpha)

# 计算所有的指标，返回到一个list
def cal_indicators(funddata,benchmark):
    sparedata = funddata.copy()
    cumunw = sparedata.iloc[-1]['net_worth']
    annualized_ret = (1+ sparedata['ret'].mean()) ** ScaleParameter - 1   # 年化收益
    annualized_vol = sparedata['ret'].std() * np.sqrt(ScaleParameter)     # 年化波动率
    odds = len(sparedata[sparedata['ret']>0]) / len(sparedata)            # 胜率
    Pnl_ratio = (sparedata[sparedata['ret']>0].mean() / sparedata[sparedata['ret']<0].mean()).values[0]  #  盈亏比
    max_dd,max_dd_start_date, max_dd_end_date = cal_max_dd_indicator(sparedata['net_worth'])           #  最大回撤/开始时间/结束时间
    annualized_downside_risk = cal_downside_risk(sparedata['ret'])                           #  年化下行风险
    VaR = cal_var(sparedata['ret'])                                                         #  VaR
    sharpe_ratio = (annualized_ret - riskFreeRate) /  annualized_vol                         #  夏普比率
    calmar_ratio = (annualized_ret - riskFreeRate) /  max_dd                                 #  卡尔马比率
    sortino_ratio = (annualized_ret - riskFreeRate) /  annualized_downside_risk              #  索提诺比率
    active_ret_300 = sparedata['ret'] - benchmark[hs300].pct_change()                        #  主动收益序列 （相对沪深300）
    active_ret_500 = sparedata['ret'] - benchmark[zz500].pct_change()                        #  主动收益序列 （相对中证500）
    information_ratio_300 = (((1+active_ret_300.mean()) ** ScaleParameter - 1) / active_ret_300.std())  \
    if active_ret_300.std() != 0  else  0                                                      #  信息比例 （相对沪深300）
    information_ratio_500 = (((1+active_ret_500.mean()) ** ScaleParameter - 1) / active_ret_500.std()) \
    if active_ret_500.std() != 0  else  0                                                      #   信息比例 （相对中证500）
    return [cumunw,annualized_ret, annualized_vol, annualized_downside_risk, odds,max_dd, max_dd_start_date, max_dd_end_date,\
            sharpe_ratio, calmar_ratio, sortino_ratio, information_ratio_300, information_ratio_500, VaR, Pnl_ratio]

# 完整的计算并输出各指标的函数，返回到一个df
def cal_indicators_to_df(fundid):
    funddata = get_fund_data(fundid,tableName =tableName)
    intersection = sorted(list(set(benchmark.index) & set(funddata.index)))      # index的交集
    benchmarkModi = benchmark.loc[intersection]
    benchmarkModi = benchmarkModi / benchmarkModi.iloc[0]                                    #  净值化
    funddata = funddata.loc[intersection]
    funddata['ret'] = funddata['net_worth'].pct_change()
    hs300nw = hs300close.loc[intersection]
    hs300nw['ret'] = zz500close.pct_change()
    zz500nw = get_benchmark(zz500).loc[intersection]
    zz500nw['ret'] = zz500nw[zz500].pct_change()
    zz500nw = zz500nw.rename(columns ={zz500:'net_worth'})
    hs300nw = hs300nw.rename(columns = {hs300:'net_worth'})
    hs300nw['net_worth'] = hs300nw['net_worth']/hs300nw['net_worth'].iloc[0]
    zz500nw['net_worth'] = zz500nw['net_worth']/zz500nw['net_worth'].iloc[0]
    resultdf = pd.DataFrame(index = [fundid,fundid+'_'+hs300,fundid+'_'+zz500],columns =columns_names, \
                        data =  [cal_indicators(funddata,benchmarkModi),cal_indicators(hs300nw,benchmarkModi),cal_indicators(zz500nw,benchmarkModi)])
    resultdf.index.name = 'fund_id'
    resultdf
    return resultdf

# 时间序列指标
def cal_rolling_indicators_to_df(fundid):
    funddata = get_fund_data(fundid,tableName =tableName)
    intersection = sorted(list(set(benchmark.index) & set(funddata.index)))      # index的交集
    benchmarkModi = benchmark.loc[intersection]
    benchmarkModi = benchmarkModi / benchmarkModi.iloc[0]                                    #  净值化
    funddata = funddata.loc[intersection]
    funddata['ret'] = funddata['net_worth'].pct_change()                                    # 收益序列
    funddata['rolling_dd'] = cal_max_dd_df(funddata['net_worth'])['max_dd']                 # 滚动最大回撤
    funddata['hs300'] = benchmarkModi['hs300']
    funddata['cumu_ret_hs300'] = funddata['hs300'] .pct_change().cumsum()                   # 沪深300的累计收益序列
    funddata['csi500'] = benchmarkModi['csi500']
    funddata['cumu_ret_csi500'] = funddata['csi500'] .pct_change().cumsum()                 # 中证500的累计收益序列
    funddata['cumulative_ret'] = funddata['ret'].cumsum()                                   # 产品的累计收益序列
    funddata = funddata.fillna(0)
    fundtosave = funddata.reset_index().set_index('fund_id')
    return fundtosave

# 近若干月收益率
def get_latest_ret(networthseries, lastestpara):
    dataseries = networthseries.iloc[-lastestpara:]
    dataseries = (dataseries / dataseries.iloc[0])
    return (dataseries.iloc[-1] - dataseries.iloc[0]) / dataseries.iloc[0]

# 今年累计收益率
def get_ret_of_this_year(funddata):
    copydata = funddata.copy()
    copydata['year'] = copydata.index.map(lambda x : x.year)
    groupgenerator = copydata.groupby('year')['net_worth']
    ret_this_year = ((groupgenerator.last() - groupgenerator.first()) / groupgenerator.first()).iloc[-1]
    return ret_this_year

# 时间序列指标
def cal_historical_performance_to_df(fundid):
    funddata = get_fund_data(fundid,tableName =tableName)
    intersection = sorted(list(set(benchmark.index) & set(funddata.index)))      # index的交集
    benchmarkModi = benchmark.loc[intersection]
    benchmarkModi = benchmarkModi / benchmarkModi.iloc[0]                                    #  净值化
    funddata = funddata.loc[intersection]
    histdf = pd.DataFrame(index = [fundid],columns = header_historical_perform,\
                         data = [[get_latest_ret(funddata['net_worth'], funddata.shape[0]),get_latest_ret(funddata['net_worth'], scale1Month),\
                                get_latest_ret(funddata['net_worth'], scale3Month),get_latest_ret(funddata['net_worth'], scale6Month),\
                                get_latest_ret(funddata['net_worth'], scale1year),get_ret_of_this_year(funddata)]])
    histdf.index.name ='fund_id'
    return histdf


# 情景分析
def cal_situation_analysis(fundid):
    funddata = get_fund_data(fundid,tableName =tableName)
    intersection = sorted(list(set(benchmark.index) & set(funddata.index)))      # index的交集
    benchmarkModi = benchmark.loc[intersection]
    benchmarkModi = benchmarkModi / benchmarkModi.iloc[0]                                    #  净值化
    funddata = funddata.loc[intersection]
    # 情景1

    try:
        data1 = funddata.loc[start1:end1]
        cumret1 = (data1['net_worth'].iloc[-1] - data1['net_worth'].iloc[0]) / data1['net_worth'].iloc[0]
        maxdd1 = cal_max_dd_indicator(data1['net_worth'])[0]
        benchmark1 = benchmarkModi.loc[start1:end1]
        cumret1_hs300 = (benchmark1['hs300'].iloc[-1] - benchmark1['hs300'].iloc[0]) / benchmark1['hs300'].iloc[0]
        maxdd1_hs300 = cal_max_dd_indicator(benchmark1['hs300'])[0]
        cumret1_csi500 = (benchmark1['csi500'].iloc[-1] - benchmark1['csi500'].iloc[0]) / benchmark1['csi500'].iloc[0]
        maxdd1_csi500 = cal_max_dd_indicator(benchmark1['csi500'])[0]
    except:
        #print 'No data between 'start1' and 'end1'!!'
        cumret1 = np.NaN
        maxdd1 = np.NaN
        cumret1_hs300 = np.NaN
        maxdd1_hs300 = np.NaN
        cumret1_csi500 = np.NaN
        maxdd1_csi500 = np.NaN
    # 情景2

    try:
        data2 = funddata.loc[start2:end2]
        cumret2 = (data2['net_worth'].iloc[-1] - data2['net_worth'].iloc[0]) / data2['net_worth'].iloc[0]
        maxdd2 = cal_max_dd_indicator(data2['net_worth'])[0]
        benchmark2 = benchmarkModi.loc[start2:end2]
        cumret2_hs300 = (benchmark2['hs300'].iloc[-1] - benchmark2['hs300'].iloc[0]) / benchmark2['hs300'].iloc[0]
        maxdd2_hs300 = cal_max_dd_indicator(benchmark2['hs300'])[0]
        cumret2_csi500 = (benchmark2['csi500'].iloc[-1] - benchmark2['csi500'].iloc[0]) / benchmark2['csi500'].iloc[0]
        maxdd2_csi500 = cal_max_dd_indicator(benchmark2['csi500'])[0]
    except:
        #print 'No dta between 'start2' and 'end2'!!'
        cumret2 = np.NaN
        maxdd2 = np.NaN
        cumret2_hs300 = np.NaN
        maxdd2_hs300 = np.NaN
        cumret2_csi500 = np.NaN
        maxdd2_csi500 = np.NaN
    # 情景3

    try:
        benchmark3 = benchmarkModi.loc[start3:end3]
        cumret3_hs300 = (benchmark3['hs300'].iloc[-1] - benchmark3['hs300'].iloc[0]) / benchmark3['hs300'].iloc[0]
        maxdd3_hs300 = cal_max_dd_indicator(benchmark3['hs300'])[0]
        cumret3_csi500 = (benchmark3['csi500'].iloc[-1] - benchmark3['csi500'].iloc[0]) / benchmark3['csi500'].iloc[0]
        maxdd3_csi500 = cal_max_dd_indicator(benchmark3['csi500'])[0]
        data3 = funddata.loc[start3:end3]
        cumret3 = (data3['net_worth'].iloc[-1] - data3['net_worth'].iloc[0]) / data3['net_worth'].iloc[0]
        maxdd3 = cal_max_dd_indicator(data3['net_worth'])[0]
    except:
        #print 'No dta between 'start3' and 'end3'!!'
        cumret3 = np.NaN
        maxdd3 = np.NaN
        cumret3_hs300 = np.NaN
        maxdd3_hs300 = np.NaN
        cumret3_csi500 = np.NaN
        maxdd3_csi500 = np.NaN
    filldata = [['2014年量化黑天鹅',start1,end1,cumret1,maxdd1],['2015年股灾',start2,end2,cumret2,maxdd2],['2016年熔断',start3,end3,cumret3,maxdd3],
           ['2014年量化黑天鹅',start1,end1,cumret1_hs300 ,maxdd1_hs300],['2015年股灾',start2,end2,cumret2_hs300,maxdd2_hs300],\
            ['2016年熔断',start3,end3,cumret3_hs300,maxdd3_hs300],['2014年量化黑天鹅',start1,end1,cumret1_csi500,maxdd1_csi500],\
            ['2015年股灾',start2,end2,cumret2_csi500,maxdd2_csi500],['2016年熔断',start3,end3,cumret3_csi500,maxdd3_csi500]]
    indexname = [fundid] * 3 + [fundid+'_'+'hs300'] * 3 + [fundid+'_'+'csi500'] * 3
    result = pd.DataFrame(index=indexname,columns =header_stress_test,data = filldata,dtype =float)
    result.index.name = 'fund_id'
    return result

if __name__ == '__main__':
    frequency = get_data_frequency(fundID, tableName=fund_info)[0][1]   # 更新频率
    if frequency == u'日度':
        ScaleParameter = 250
        scale1Month = 22
        scale3Month = 63
        scale6Month = 125
        scale1year = 250
    elif frequency == u'周度':
        ScaleParameter = 50
        scale1Month = 4
        scale3Month = 12
        scale6Month = 24
        scale1year = 50
    elif frequency == u'月度':
        ScaleParameter = 12
        scale1Month = 1
        scale3Month = 3
        scale6Month = 6
        scale1year = 12

    fundtype = str(int(get_fund_type(fundID, tableName=fund_type_table)['stype_code'].iloc[0]))  # 确保是可以读取的形式
    fundname = get_fund_type_name(u'股票策略', tableName=fund_type_table)
    indextypemaptable = get_type_index_table()
    index = indextypemaptable.loc[fundtype]['index_id']  # 基金对应的指数代码
    indexnetworth = get_index(index, tableName=index_table)  # 基金对应的指数的净值数据

    # globals
    benchmark = get_benchmark(hs300).merge(get_benchmark(zz500), how='inner', left_index=True,
                                           right_index=True)  # 沪深300 /中证500的日指数

    hs300close = get_benchmark(hs300)
    zz500close = get_benchmark(zz500)

    fund_list_stock = ['JR000001', 'JR000002', 'JR000003', 'JR000004', 'JR000008']
    fund_list_bond = ['JR000005', 'JR000056', 'JR000065', 'JR000104', 'JR000108']

    # test
    indicatorsDF = pd.DataFrame()
    rollingIndiDF = pd.DataFrame()
    histperfDF = pd.DataFrame()
    sitAnalysisDF = pd.DataFrame()
    for fund in fund_list_bond + fund_list_stock:
        print fund
        try:
            indicatorsDF = indicatorsDF.append(cal_indicators_to_df(fund))  # 基金及其对应的benchmark指标
            rollingIndiDF = rollingIndiDF.append(cal_rolling_indicators_to_df(fund))  # 基金的时间序列指标
            histperfDF = histperfDF.append(cal_historical_performance_to_df(fund))  # 基金的历史指标
            sitAnalysisDF = sitAnalysisDF.append(cal_situation_analysis(fund))    # 情景分析
        except:
            print 'No data in Data Base!'
            continue

    indicatorsDF = indicatorsDF.round(4)
    rollingIndiDF = rollingIndiDF.round(4)
    histperfDF = histperfDF.round(4)
    sitAnalysisDF = sitAnalysisDF.round(4)

    # 插入到数据库

    import sqlalchemy.engine.url as url
    from sqlalchemy import create_engine

    engine_url = url.URL(drivername='mysql', host='119.254.153.20', port=13311, username='tai', password='tai2015',
                         database='PrivateEquityFund_W',
                         query={'charset': 'utf8'})  # 创建可插入中文的引擎
    db = create_engine(engine_url, encoding='utf-8')  # 注意encoding选项（mysql默认是latin,这里要改成utf-8）

    # indicatorsDF.reset_index().to_sql(name='fund_indicators_w',con=db, if_exists='replace',index=False)
    # rollingIndiDF.reset_index().to_sql(name='fund_rolling_indicators_w',con=db , if_exists='replace',index=False)
    # histperfDF.reset_index().to_sql(name='fund_historical_performance_w',con=db , if_exists='replace',index=False)
    #sitAnalysisDF.reset_index().to_sql(name='fund_situation_analysis_w',con=db, if_exists='replace',index=False)
    cnx.close()

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
import json
import mysql.connector


#  读取数据库的指针设置
with open('conf.json', 'r') as fd:
    conf = json.load(fd)
src_db = mysql.connector.connect(**conf['src_db'])

#  一些常量
riskFreeRate = 0.02  # 无风险利率
varThreshold =0.05   # 5%VAR阈值
scaleParameter = 50  # 一年50周

# 表名
index_data_table = 'fund_weekly_index'  # index时间序列数据
index_name_table = 'index_id_name_mapping'
type_index_table = 'index_stype_code_mapping' # 表格名称-不同基金种类对应的指数

# 私募指数基金分类表格对应（只需要跑一次）
def get_type_index_table(tableName = type_index_table):

    try:
        #sql_query='select id,name from student where  age > %s'
        cursor = src_db .cursor()
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

# 私募指数名称及ID分类表格对应（只需要跑一次）
def get_index_table(tableName = index_name_table):

    try:
        #sql_query='select id,name from student where  age > %s'
        cursor = src_db .cursor()
        sql = "select * from %s" % (tableName)
        cursor.execute(sql)
        result = cursor.fetchall()
    finally:
        pass
    #pdResult = dict(result)
    pdResult = pd.DataFrame(result)
    pdResult = pdResult.dropna(axis=0)
    pdResult.columns = [i[0] for i in cursor.description]
    pdResult.set_index('index_id',inplace=True)
    return pdResult

# 私募指数净值的时间序列
def get_index(index, tableName=index_data_table):
    try:
        # sql_query='select id,name from student where  age > %s'
        cursor = src_db.cursor()
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

# 按季度分类
def byseasons(x):
    if 1<=x.month<=3:
        return str(x.year)+'_'+str(1)
    elif 4<= x.month <=6:
        return str(x.year)+'_'+str(2)
    elif 7<= x.month <=9:
        return str(x.year)+'_'+str(3)
    else:
        return str(x.year)+'_'+str(4)

# 计算最大回撤，最大回撤开始结束时间
def cal_max_dd_indicator(networthSeries):
    maxdd = pd.DataFrame(index = networthSeries.index, data=None, columns =['max_dd','max_dd_start_date','max_dd_end_date'],dtype = float)
    maxdd.iloc[0] = 0
    maxdd.is_copy = False
    for date in networthSeries.index[1:]:
        maxdd.loc[date] = [1 - networthSeries.loc[date] / networthSeries.loc[:date].max(),networthSeries.loc[:date].idxmax(),date]
        #maxdd[['max_dd_start_date','max_dd_end_date']].loc[date] = [[networthSeries.loc[:date].idxmax(),date]]
        #maxdd['max_dd_start_date'].loc[date] = networthSeries.loc[:date].idxmax()
    return maxdd['max_dd'].max(), maxdd.loc[maxdd['max_dd'].idxmax]['max_dd_start_date'],maxdd.loc[maxdd['max_dd'].idxmax]['max_dd_end_date']

# 计算最大回撤（每季度），输入为dataframe，输出也为dataframe
def cal_maxdd_by_season(df):
    seasonList = sorted(list(set(df['season'].values)))
    maxdd_dict = {}
    for season in seasonList:
        temp = df[df['season'] == season]
        maxdd_dict[season] = np.round(cal_max_dd_indicator(temp['net_worth'])[0],4)
    maxdd_df = pd.DataFrame([maxdd_dict]).T
    maxdd_df.columns =[df['index'].iloc[0]]
    maxdd_df.index.name = 'season'
    return maxdd_df

# 计算最大回撤（每年）,输入为dataframe，输出也为dataframe
def cal_maxdd_by_year(df):
    seasonList = sorted(list(set(df['year'].values)))
    maxdd_dict = {}
    for season in seasonList:
        temp = df[df['year'] == season]
        maxdd_dict[season] = np.round(cal_max_dd_indicator(temp['net_worth'])[0],4)
    maxdd_df = pd.DataFrame([maxdd_dict]).T
    maxdd_df.columns =[df['index'].iloc[0]]
    maxdd_df.index.name = 'year'
    return maxdd_df


# 准备数据原始dataframe
def get_count_data(cnx):
    cursor = cnx.cursor()
    sql = "select fund_id,foundation_date,fund_type_strategy from fund_info"
    cursor.execute(sql)
    result = cursor.fetchall()
    df = pd.DataFrame(result)
    df.columns = ['fund_id', 'found_date', 'strategy']

    sql = "select type_id, strategy from index_type_mapping"
    cursor.execute(sql)
    result = cursor.fetchall()
    meg = pd.DataFrame(result)
    meg.columns = ['type_id', 'strategy']

    # 数据清理
    df = df.dropna()
    df = df[df['strategy'] != u'']
    # 合并对应表
    df = pd.merge(df, meg)
    # 加年份列
    df['year'] = [str(i.year) for i in df['found_date']]
    # 加月份列
    df['month'] = [str(i.year) + '_' + str(i.month) for i in df['found_date']]
    return df.drop('strategy', axis=1)


# 得到按年份分类统计，输出 dataframe
def get_ann_fund(df):
    temp = df.groupby(['type_id', 'year'])['fund_id'].count().to_frame()  # 分类转dataframe
    temp = pd.pivot_table(temp, values='fund_id', index='year', columns=['type_id'])
    temp['Type_0'] = df.groupby(['year'])['fund_id'].count().to_frame()['fund_id']  # 添加全市场数据
    temp.sort_index(axis=0)
    temp.sort_index(axis=1, inplace=True)
    return temp


# 得到按月份分类统计， 输出dataframe
def get_month_fund(df):
    temp = df.groupby(['type_id', 'month'])['fund_id'].count().to_frame()  # 分类转dataframe
    temp = pd.pivot_table(temp, values='fund_id', index=['month'], columns=['type_id'])
    temp['Type_0'] = df.groupby(['month'])['fund_id'].count().to_frame()['fund_id']  # 添加全市场数据
    temp.sort_index(axis=0)
    temp.sort_index(axis=1, inplace=True)
    return temp


# 准备数据原始dataframe
def get_org_count(cnx):
    cursor = cnx.cursor()
    sql = "SELECT org_id, found_date FROM PrivateEquityFund.org_info WHERE org_category LIKE '4%'"
    cursor.execute(sql)
    result = cursor.fetchall()
    df = pd.DataFrame(result)
    df.columns = ['org_id', 'found_date']

    # 数据清理
    df = df.dropna()

    # 加年份列
    df['year'] = [str(i.year) for i in df['found_date']]
    # 加月份列
    df['month'] = [str(i.year) + '_0' + str(i.month) if i.month < 10 else str(i.year) + '_' + str(i.month) for i in
                   df['found_date']]

    return df


# 得到按年份分类统计，输出 dataframe
def get_ann_org(df):
    temp = df.groupby(['year'])['org_id'].count().to_frame()  # 分类转dataframe
    temp.sort_index(axis=0)
    return temp


# 得到按月份分类统计， 输出dataframe
def get_month_org(df):
    temp = df.groupby(['month'])['org_id'].count().to_frame()  # 分类转dataframe
    temp.sort_index(axis=0)
    return temp


if __name__ == '__main__':
    # 计算季度指标
    maxddbyseason = pd.DataFrame()  # 季度最大回撤
    retbyseason = pd.DataFrame()  # 季度收益
    stdbyseason = pd.DataFrame()  # 极度标准差
    sharpebyseason = pd.DataFrame()  # 季度夏普

    # 计算年度指标
    maxddbyyear = pd.DataFrame()  # 年度最大回撤
    retbyyear = pd.DataFrame()  # 年度收益
    stdbyyear = pd.DataFrame()  # 年度标准差
    sharpebyyear = pd.DataFrame()  # 年度夏普

    indexIDdf = get_index_table()   # 获取指数表格
    for index in indexIDdf.index:
        # 季度指标
        indexdf = get_index(index, tableName=index_data_table)
        indexdf['pnl'] = indexdf['net_worth'].pct_change()
        indexdf['season'] = indexdf.index.map(byseasons)
        indexdf['year'] = indexdf.index.map(lambda x: x.year)
        maxdd_season = cal_maxdd_by_season(indexdf)
        maxddbyseason = maxddbyseason.merge(maxdd_season, how='outer', left_index=True, right_index=True)

        indexbyseason = indexdf.groupby('season')['pnl']
        ret_season = (indexbyseason.mean() + 1) ** scaleParameter - 1  # 年化收益(季度)
        std_season = np.sqrt(scaleParameter) * indexbyseason.std()  # 年化标准差（季度）
        sharpe_season = (ret_season - riskFreeRate) / std_season  # 夏普比率（季度）
        ret_season = pd.DataFrame(ret_season).round(4)  # series 转换为 dataframe
        ret_season.columns = [indexdf['index'].iloc[0]]  # 添加列名
        std_season = pd.DataFrame(std_season).round(4)
        std_season.columns = [indexdf['index'].iloc[0]]
        sharpe_season = pd.DataFrame(sharpe_season).round(4)
        sharpe_season.columns = [indexdf['index'].iloc[0]]

        retbyseason = retbyseason.merge(ret_season, how='outer', left_index=True, right_index=True)
        stdbyseason = stdbyseason.merge(std_season, how='outer', left_index=True, right_index=True)
        sharpebyseason = sharpebyseason.merge(sharpe_season, how='outer', left_index=True, right_index=True)

        # 年度指标
        maxdd_year = cal_maxdd_by_year(indexdf)
        maxddbyyear = maxddbyyear.merge(maxdd_year, how='outer', left_index=True, right_index=True)

        indexbyyear = indexdf.groupby('year')['pnl']
        ret_year = (indexbyyear.mean() + 1) ** scaleParameter - 1  # 年化收益(季度)
        std_year = np.sqrt(scaleParameter) * indexbyyear.std()  # 年化标准差（季度）
        sharpe_year = (ret_year - riskFreeRate) / std_year  # 夏普比率（季度）
        ret_year = pd.DataFrame(ret_year).round(4)  # series 转换为 dataframe
        ret_year.columns = [indexdf['index'].iloc[0]]  # 添加列名
        std_year = pd.DataFrame(std_year).round(4)
        std_year.columns = [indexdf['index'].iloc[0]]
        sharpe_year = pd.DataFrame(sharpe_year).round(4)
        sharpe_year.columns = [indexdf['index'].iloc[0]]

        retbyyear = retbyyear.merge(ret_year, how='outer', left_index=True, right_index=True)
        stdbyyear = stdbyyear.merge(std_year, how='outer', left_index=True, right_index=True)
        sharpebyyear = sharpebyyear.merge(sharpe_year, how='outer', left_index=True, right_index=True)

    # 统计发行的基金数量信息
    countfund = get_count_data(src_db)   # 准备数据
    countfundbyyear = get_ann_fund(countfund)   # 发行基金数量统计（年）
    countfundbymonth = get_month_fund(countfund)  # 发行基金数量统计 （月）

    # 统计成立的私募公司
    countorg = get_org_count(src_db)   # 准备数据
    countorgbyyear = get_ann_org(countorg)   # 成立私募数量统计（年）
    countorgbymonth = get_month_org(countorg)  # 成立私募数量统计 （月）




    # 插入数据库
    from sqlalchemy import create_engine

    # MSText(length=255)
    db_engine = create_engine(
        'mysql://{0}:{1}@{2}:{3}/{4}'.format('tai', 'tai2015', '119.254.153.20', 13311, 'PrivateEquityFund_W',
                                             encoding='utf-8'))
    # 季度指标
    #maxddbyseason.reset_index().to_sql(name='index_maxdd_by_season', con=db_engine, if_exists='replace', index=False)
    #retbyseason.reset_index().to_sql(name='index_return_by_season', con=db_engine, if_exists='replace', index=False)
    #stdbyseason.reset_index().to_sql(name='index_std_by_season', con=db_engine, if_exists='replace', index=False)
    #sharpebyseason.reset_index().to_sql(name='index_sharpe_by_season', con=db_engine, if_exists='replace', index=False)
    # 年度指标
    #maxddbyyear.reset_index().to_sql(name='index_maxdd_by_year', con=db_engine, if_exists='replace', index=False)
    #retbyyear.reset_index().to_sql(name='index_return_by_year', con=db_engine, if_exists='replace', index=False)
    #stdbyyear.reset_index().to_sql(name='index_std_by_year', con=db_engine, if_exists='replace', index=False)
    #sharpebyyear.reset_index().to_sql(name='index_sharpe_by_year', con=db_engine, if_exists='replace', index=False)

    # 上传年度数据
    #countfundbyyear.reset_index().to_sql(name='fund_count_annual', con=db_engine, if_exists='replace', index=False)
    # 上传月度数据
    #countfundbymonth.reset_index().to_sql(name='fund_count_month', con=db_engine, if_exists='replace', index=False)

    # 上传年度数据
    # countorgbyyear.reset_index().to_sql(name='fund_count_annual', con=db_engine, if_exists='replace', index=False)
    # 上传月度数据
    # countorgbymonth.reset_index().to_sql(name='fund_count_month', con=db_engine, if_exists='replace', index=False)
    src_db.close()


#!/Tsan/bin/python
# -*- coding: utf-8 -*-

# Libraries To Use
from __future__ import division
import numpy as np
import pandas as pd
from datetime import datetime,time,date
from collections import OrderedDict


# Import My own library for factor testing
from SingleFactorTest import factorFilterFunctions as ff





#----------------------------------------------------------------------
def formatNumber(n):
    """格式化数字到字符串"""
    rn = round(n, 4)        # 保留三位小数
    return format(rn, ',')  # 加上千分符

# 下单/成交单的类
class OrderData(object):
    def __init__(self, datetime, stkID, volume, price, direction):
        self.datetime = datetime
        self.stkID = stkID
        self.volume = volume
        self.price = price
        self.direction = direction

# 持仓类
class holdingData(object):
    def __init__(self, volume, averageCost):
        #self.stkID = stkID
        self.volume = volume
        self.averageCost = averageCost


# backtesting class
class StkBacktesting(object):
    """开盘时（9：30）按找开盘价下单交易，到收盘时当日持仓不会变。
        收盘时（15：00）按照收盘价结算当日持仓收益。
        收益分为两个部分：1.交易时产生的pnl，在交易时结算（9：30）。
                        2. 持仓产生的pnl，在当天收盘时结算（15:00）。

        交易时注意要先卖后买。
        注意本框架计算收益用的是股票成本价变动法，即当发生买卖时会将新的交易信息归入持仓成本里（重新计算股票成本价），
        所以只要股票目前有持仓，则持仓收益即为股票到目前的总收益。（中间卖完过除外）。
        交易收益字典里只有当天被完全被卖出的股票（没有持仓），会被添加到当天的交易pnl字典里。
        """

    # ----------------------------------------------------------------------
    def __init__(self, path):

        self.path = path

        self.dataDict = {}

        self.commisionRate = 0.003  # 手续费（买卖都有）
        self.stampTax = 0.001  # 印花税（只有卖的时候有）

        self.initCap = 1000000
        self.availableCashNow = self.initCap
        # self.marketValue = 0

        self.winRateCal = []  # 添加每一笔卖出交易的盈利用来计算胜率


        self.currentPositionDict = {}  # 最新持仓，key为stkID，value为持仓数量和平均持仓成本

        self.allOrdersDict = {}  # 最新持仓，key为日期，value为list,为当天的下单,list里的每个元素为order
        self.tradingPnlDict = {}  # 交易时产生的收益（亏损），key为date，value为dictionary,到当日为止该股票交易的累积收益
        self.holdingPnlDict = {}  # 持仓产生的收益（市值变动）,key为date, value为dictionary，为截止当日为止持仓股票的市值变动
        self._allCurrentPositionDict = {}  # 当日持仓, key为date, value为dictionary，为当日持仓股票
        # self.stkPnlOneDay = {}    # 某天产生的所有收益（按股票分开）,key为date, value为dictionary,为当日股票的pnl(持仓+卖出)

        #self.limitOrderCount = 0
        #self.orderCount = 0


        self.totalTradingAmountDict = OrderedDict()  # 总交易额字典,key为日期，value为当天成交值
        self.totalCommisionDict = OrderedDict()  # 总交易额字典,key为日期，value为当天总手续费
        self.totalMarketValueDict = OrderedDict()  # 总市值字典,key为日期，value为当天市值
        self.availableCashDict = OrderedDict()  # 可用资金字典,key为日期，value为当天可用资金
        self.limitOrderDict = OrderedDict()
        # self.orderbyDay = OrderedDict()  # 下单的每日成交字典,key为日期。value为当日下的的所有单的list。
        self.tradeDict = OrderedDict()  # 撮合成交后的每日成交字典,key为日期。value为当日成交的所有单的list。

        # self.backTestingDateList = []

    # ----------------------------------------------------------------------
    def setInitCap(self, initCap):
        """设置初始资金"""
        self.initCap = initCap

    # ----------------------------------------------------------------------
    def setCommisionRate(self, commisionRate):
        """设置手续费率"""
        self.commisionRate = commisionRate

    # ----------------------------------------------------------------------
    def setCommisionRate(self, stampTax):
        """设置印花费税"""
        self.stampTax = stampTax

    # ----------------------------------------------------------------------
    def makeOrder(self, datetime, stkID, amount, price, direction):

        order = OrderData(datetime, stkID, amount, price, direction)
        return order


    # ----------------------------------------------------------------------
    def setBackTestingPeriod(self, startTime, endTime):
        """设置回测起始时间"""
        try:
            self.backTestingDateList = self.dataDict['adjOpen'].loc[startTime:endTime].index
        except KeyError:
            print 'No available data! Plz feed data first!'

    # ----------------------------------------------------------------------
    def setInitialPeriod(self, initialstartTime, initialendTime):
        """设置回测起始时间"""
        for name, factor in self.dataDict.iteritems():
            self.dataDict[name] = factor.loc[initialstartTime:initialendTime]

    # ----------------------------------------------------------------------

    def broadCastingData(self):
        """构造一个回测期间*购买股票的二维表"""
        self.totalInfodf = pd.DataFrame(index=self.backTestingDateList,
                                       columns=pd.DataFrame.from_dict(self.holdingPnlDict, orient='index').columns)

    # ----------------------------------------------------------------------
    def addData(self, key, filename):
        """添加数据
        key: STRING ,the key of the data in datadict.
        filename: STRING ,the name of the data file exist in the path,,
        Note that the postfix 'h5' is  necessary here.
        """
        self.dataDict[key] = ff.readh5data(self.path, filename)

    # ----------------------------------------------------------------------
    def getCurrentPosition(self):
        """获取当前仓位信息，主要用处是在策略内运行时获取现在的仓位"""
        return self.currentPositionDict

    # ----------------------------------------------------------------------
    def getAvailableCashNow(self):
        """获取当前可用资金，主要用处是在策略内运行时获取现在可用资金"""
        return self.availableCashNow

    # ----------------------------------------------------------------------
    def getAllPosition(self):
        """获取所有日期的仓位信息，用途
        1.风险因子暴露分析；
        2.行业和大小盘收益分析；
        请在全部运行完后再使用"""
        return self._allCurrentPositionDict

    # ----------------------------------------------------------------------
    def runStrategy(self):
        pass

    # ----------------------------------------------------------------------
    def calTotalAsset(self):
        """计算总收益"""
        self.totalAsset = pd.concat([pd.Series(self.totalMarketValueDict), pd.Series(self.availableCashDict)],
                                    axis=1)  # 合成dataframe
        self.totalAsset.columns = ['MarketValue', 'AvailableCash']
        self.totalAsset = self.totalAsset.reindex(self.backTestingDateList)
        self.totalAsset.iloc[0] = [0, self.initCap]
        self.totalAsset = self.totalAsset.fillna(method='ffill')
        # self.totalAsset.sum(axis=1).plot()
        return self.totalAsset

    # ----------------------------------------------------------------------
    def calStkPnlDistribution(self):
        """计算各股票的pnl收益，计算出一个日期*股票的二维表，值为当天的个股收益"""
        holdingPnldf = pd.DataFrame.from_dict(self.holdingPnlDict, orient='index').fillna(0).sort_index('inplace=True')
        tradingPnldf = pd.DataFrame.from_dict(self.tradingPnlDict, orient='index').reindex_like(holdingPnldf).fillna(
            0).sort_index('inplace=True')
        return holdingPnldf + tradingPnldf

    # ----------------------------------------------------------------------
    def calTotalMarketValue(self, closePrice):
        """计算当日市值"""
        try:
            volumeSeries = pd.Series({stkID: holding.volume for stkID, holding in self.currentPositionDict.iteritems()})
            totalMV = (volumeSeries * closePrice[volumeSeries.index.tolist()]).sum()
        except:
            totalMV = 0
        return totalMV

    # ----------------------------------------------------------------------
    def calTurnOverRatio(self):
        """计算换手率"""
        nominator = pd.Series(self.totalTradingAmountDict).reindex(self.backTestingDateList).fillna(0)
        denominator = pd.Series(self.totalMarketValueDict).reindex(self.backTestingDateList).fillna('ffill').dropna()
        return (nominator[denominator.index] / denominator).mean()

    # ----------------------------------------------------------------------
    def crossOrder(self, date):
        """给定日期进行撮合及pnl/availableCash/marketValue的更新"""
        try:
            openPrice = self.dataDict['adjOpen'].loc[date]
            closePrice = self.dataDict['adjClose'].loc[date]
            trueVolume = self.dataDict['volume'].loc[date]
        except KeyError:
            print date
            print 'No available data! Plz feed data first!'

        tradingList = []
        holdingPnlDict = {}  # 持仓收益
        tradingPnlToday = {}  # 交易收益
        # cashNetInThisDay = 0  # 资金净流入
        tradingAmountToday = 0  # 当日成交额
        tradingCommisionToday = 0  # 当日总手续费
        for order in self.allOrdersDict[date]:
            if not np.isnan(openPrice[order.stkID]):
                crossPrice = openPrice[order.stkID]  # 最优价格
                # dealPrice = crossPrice    # 成交价格
                dealVolume = min(order.volume, 0.05 * trueVolume[order.stkID])  # 最多成交当天交易量的5%
                trade = OrderData(order.datetime, order.stkID, dealVolume, crossPrice, order.direction)
                #trade.datetime = order.datetime
                #trade.stkID = order.stkID
                #trade.volume = dealVolume
                #trade.price = crossPrice
                #trade.direction = order.direction

                # 计算该次交易的净现金流
                stkNetcash = - trade.volume * trade.price * (1 + self.commisionRate) if trade.direction == 1 else \
                    (1 - self.commisionRate - self.stampTax) * trade.volume * trade.price

                # 如果目前可用资金太少
                if self.availableCashNow < - stkNetcash:
                    continue

                # 添加到tradingList里
                tradingList.append(trade)

                # 当日资金净流入
                # cashNetInThisDay += stkNetcash

                # 当日成交额
                tradingAmountToday += trade.volume * trade.price

                # 总可用资金
                self.availableCashNow += stkNetcash

                # 个股成交手续费计算
                trade.commisionCost = self.commisionRate * trade.volume * trade.price if trade.direction == 1 else \
                    (self.commisionRate + self.stampTax) * trade.volume * trade.price

                # 手续费计算
                tradingCommisionToday += trade.commisionCost

                # 若持仓字典中已存在此股票ID
                if trade.stkID in self.currentPositionDict:
                    # 更新持仓量
                    holding = holdingData(0, 0)
                    holding.volume = self.currentPositionDict[trade.stkID].volume + trade.direction * trade.volume

                    # 此笔为卖出交易时
                    if trade.direction == -1:
                        # 添加卖出交易的盈利至胜率计算字典
                        thistradepnl = (trade.price - self.currentPositionDict[
                                trade.stkID].averageCost) * trade.volume - trade.commisionCost
                        self.winRateCal.append(thistradepnl)
                        # 若交易后持仓量小于等于0，则删除此股票持仓信息
                        if holding.volume <= 0:
                            # 卖出后无持仓，则添加此笔交易到盈利字典
                            tradingPnlToday[trade.stkID] = thistradepnl
                            del self.currentPositionDict[trade.stkID]
                            continue

                    # 更新持仓成本
                    holding.averageCost = (self.currentPositionDict[trade.stkID].averageCost * self.currentPositionDict[
                        trade.stkID].volume +
                        trade.commisionCost + trade.direction * (trade.price * trade.volume)) / holding.volume
                    #
                    self.currentPositionDict[trade.stkID] = holding

                # 若不存在,此时只有在买入时才会添加持仓信息,相当于新买入。
                else:
                    if trade.direction == 1:
                        holding = holdingData(0, 0)
                        holding.volume = trade.direction * trade.volume
                        holding.averageCost = trade.price * (1 + self.commisionRate)

                        self.currentPositionDict[trade.stkID] = holding

                    else:
                        print 'Short is not allowed in Stock Market of China!!'

                # 计算总的持仓收益
                holdingPnlDict[trade.stkID] = (closePrice[trade.stkID] - self.currentPositionDict[
                    trade.stkID].averageCost) * self.currentPositionDict[trade.stkID].volume

            else:
                continue

        # 计算当日交易完后的总市值（收盘时）
        self.totalMarketValueDict[date] = self.calTotalMarketValue(closePrice)

        # 计算当日交易完后的可用资金
        self.availableCashDict[date] = self.availableCashNow

        # 添加当日总交易额至字典
        self.totalTradingAmountDict[date] = tradingAmountToday

        # 添加当日总交易额至字典
        self.totalCommisionDict[date] = tradingCommisionToday

        # 保存当日持仓Pnl至持仓Pnl信息字典
        self.holdingPnlDict[date] = holdingPnlDict

        # 当日交易Pnl
        self.tradingPnlDict[date] = tradingPnlToday

        # 保存当日持仓字典
        self._allCurrentPositionDict[date] = self.currentPositionDict

        # 保存当日成交信息至成交字典
        self.tradeDict[date] = tradingList

    # ----------------------------------------------------------------------
    def output(self, content):
        """输出内容"""
        print str(datetime.now()) + "\t" + content

    # ----------------------------------------------------------------------
    def showBackTestingResult(self):
        """计算各指标，并作图"""

        totalcommision = pd.Series(self.totalCommisionDict).sum()
        turnoverrate = self.calTurnOverRatio()
        winrate = pd.Series(self.winRateCal)
        winrate = winrate[winrate > 0].shape[0] / winrate .shape[0]
        totaldf = self.calTotalAsset()  # 计算总资产
        networth = totaldf.sum(axis=1) / totaldf.sum(axis=1).iloc[0]
        annualizedRet = (networth.iloc[-1]) ** (12 / networth.shape[0]) - 1
        annualizedvol = np.sqrt(12) * networth.pct_change().iloc[1:].std()
        sharpe = annualizedRet / annualizedvol
        rollingdd = (1 - (networth / networth.expanding().max()))
        maxdd = rollingdd.max()

        marketValue = totaldf.divide(totaldf.sum(axis=1), axis=0)

        self.output(u'期末净值：\t%s' % formatNumber(networth.iloc[-1]))
        self.output(u'总盈亏：\t%s' % formatNumber(totaldf.sum(axis=1).iloc[-1] - self.initCap))
        self.output(u'年化收益率：\t%s' % formatNumber(annualizedRet))
        self.output(u'年化波动率：\t%s' % formatNumber(annualizedvol))
        self.output(u'夏普比率：\t%s' % formatNumber(sharpe))
        self.output(u'最大回撤: \t%s' % formatNumber(maxdd))
        self.output(u'胜率: \t%s' % formatNumber(winrate))
        self.output(u'换手率: \t%s' % formatNumber(turnoverrate))
        self.output(u'总手续费: \t%s' % formatNumber(totalcommision))

        # 绘图
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        try:
            import seaborn as sns  # 如果安装了seaborn则设置为白色风格
            sns.set_style('whitegrid')
        except ImportError:
            pass

        colorpalette = sns.color_palette("Paired", 2)
        fig = plt.figure(figsize=(16, 10))
        # 净值曲线
        pCapital = plt.subplot(4, 1, 1)
        pCapital.set_ylabel("networth")
        pCapital.plot(networth.index, networth.values, color='r', lw=0.8)
        plt.title('Networth Chart')

        # 最大回撤曲线
        pDD = plt.subplot(4, 1, 2)
        pDD.set_ylabel("maxdd")
        pDD.bar(rollingdd.index, rollingdd.values, color='g')
        plt.title('Max Draw Down')

        # 股票市值占比曲线

        '''
        pMV = plt.subplot(4, 1, 3)
        pMV.set_ylabel("percentage")
        pMV.legend(['MarketValue', 'AvailableCash'], loc='upper right')
        pMV.stackplot(marketValue.index, marketValue['MarketValue'], marketValue['AvailableCash'], colors=[colorpalette[0], colorpalette[1]])
        labels = ['MarketValue', 'AvailableCash']

        # legend 属性和stackplot不兼容，因此要额外画俩小方块作为标识
        p1 = Rectangle((0, 0), 1, 1, fc=colorpalette[0])
        p2 = Rectangle((0, 0), 1, 1, fc=colorpalette[1])
        plt.legend([p1, p2], labels)
        plt.title('Cash/MarketValue Ratio')
        '''

        pMV = plt.subplot(4, 1, 3)
        pMV.stackplot(marketValue.index, marketValue['MarketValue'], marketValue['AvailableCash'],
                      labels=['MarketValue', 'AvailableCash'])
        pMV.legend(loc='upper right')
        # plt.margins(0,0)
        plt.title('Cash/MarketValue Ratio')

        plt.tight_layout()
        plt.show()




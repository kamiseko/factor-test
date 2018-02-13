#!/Tsan/bin/python
# -*- coding: utf-8 -*-

# Libraries To Use
from __future__ import division
import numpy as np
import pandas as pd
from datetime import datetime,time,date
from collections import OrderedDict
import copy


# Import My own library for factor testing
from SingleFactorTest import factorFilterFunctions as ff




Open = 'open'
Cover = 'cover'

#----------------------------------------------------------------------
def formatNumber(n):
    """格式化数字到字符串"""
    rn = round(n, 4)        # 保留三位小数
    return format(rn, ',')  # 加上千分符

# 下单/成交单的类
class OrderData(object):
    def __init__(self, datetime, stkID, volume, price, direction, offset):
        self.datetime = datetime
        self.stkID = stkID
        self.volume = volume
        self.price = price
        self.direction = direction
        self.offset = offset

# 持仓类
class holdingData(object):
    def __init__(self, volume, averageCost, direction):
        #self.stkID = stkID
        self.volume = volume
        self.averageCost = averageCost
        self.direction = direction


# backtesting class
class FutureBacktesting(object):
    """开盘时（9：30）按照开盘价下单交易，到收盘时当日持仓不会变。
        收盘时（15：00）按照收盘价结算当日持仓收益。
        收益分为两个部分：1.交易时产生的pnl，在交易时结算（9：30）。
                        2. 持仓产生的pnl，在当天收盘时结算（15:00）。

        交易时注意要先卖后买。
        注意本框架计算收益用的是股票成本价变动法，即当发生买卖时会将新的交易信息归入持仓成本里（重新计算股票成本价），
        所以只要股票目前有持仓，则持仓收益即为股票到目前的总收益。（中间股票持仓全部卖完过除外）。
        交易收益字典里只有当天被完全被卖出的股票（没有持仓），会被添加到当天的交易pnl字典里。
        """

    # ----------------------------------------------------------------------
    def __init__(self, path):

        self.path = path

        self.dataDict = {}

        self.commisionRate = 0.0003  # 手续费
        self.stampTax = 0.001  # 印花税（只有卖的时候有）

        self.marginRatio = {}  # 保证金字典
        self.multiplier = {}  # 合约乘数字典
        self.tickSize = {}  # 合约最小变动单位

        self.initCap = 1000000
        self.availableCashNow = self.initCap
        # self.marketValue = 0

        self.winRateCal = []  # 添加每一笔卖出交易的盈利用来计算胜率


        self.currentPositionDict = {}  # 最新持仓，key为stkID，value为持仓数量和平均持仓成本

        self.allOrdersDict = {}  # 最新持仓，key为日期，value为list,为当天的下单,list里的每个元素为order
        self.tradingPnlDict = {}  # 交易时产生的收益（亏损），key为date，value为dictionary,到当日为止该股票交易的累积收益
        self.holdingPnlDict ={}  # 持仓产生的收益（市值变动）,key为date, value为dictionary，为截止当日为止持仓股票的市值变动
        self._allCurrentPositionDict = {}  # 当日持仓, key为date, value为dictionary，为当日持仓股票


        #self.limitOrderCount = 0
        #self.orderCount = 0


        self.totalTradingAmountDict = OrderedDict()  # 总交易额字典,key为日期，value为当天成交值
        self.totalCommisionDict = OrderedDict()  # 总交易额字典,key为日期，value为当天总手续费
        self.totalMarketValueDict = OrderedDict()  # 总市值字典,key为日期，value为当天市值
        self.availableCashDict = OrderedDict()  # 可用资金字典,key为日期，value为当天可用资金
        self.limitOrderDict = OrderedDict()
        self.tradeDict = OrderedDict()  # 撮合成交后的每日成交字典,key为日期。value为当日成交的所有单的list。

        # self.backTestingDateList = []

    # ----------------------------------------------------------------------
    def setInitCap(self, initCap):
        """设置初始资金"""
        self.initCap = initCap
        self.availableCashNow = initCap

    # ----------------------------------------------------------------------
    def setCommisionRate(self, commisionRate):
        """设置手续费率"""
        self.commisionRate = commisionRate

    # ----------------------------------------------------------------------
    def setCommisionRate(self, stampTax):
        """设置印花费税"""
        self.stampTax = stampTax

    # ----------------------------------------------------------------------
    def setMarginRatio(self, marginRatio):
        """设置各合约保证金, 注意文件为一个dictionary"""
        self.marginRatio = marginRatio

    # ----------------------------------------------------------------------
    def setMultiplier(self, multiplier):
        """设置各合约乘数, 注意文件为一个dictionary"""
        self.multiplier = multiplier

    # ----------------------------------------------------------------------
    def setTickSize(self, tickSize):
        """设置各合约最小变动单位, 注意文件为一个dictionary"""
        self.tickSize = tickSize


    # ----------------------------------------------------------------------
    def setContractSize(self, contractSize):
        """设置各品种合约大小, 注意文件为一个dataframe"""
        self.contractSize = contractSize

    # ----------------------------------------------------------------------
    def setPriceTick(self, priceTick):
        """设置各品种价格最小变动, 注意文件为一个dataframe"""
        self.priceTick = priceTick

    # ----------------------------------------------------------------------
    def makeOrder(self, datetime, stkID, amount, price, direction, offset):

        order = OrderData(datetime, stkID, amount, price, direction, offset)
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
        """设置初始化起始时间"""
        for name, factor in self.dataDict.iteritems():
            self.dataDict[name] = factor.loc[initialstartTime:initialendTime]

    # ----------------------------------------------------------------------

    def broadCastingData(self):
        """构造一个回测期间*购买股票数量的二维表"""
        self.totalInfodf = pd.DataFrame(index=self.backTestingDateList,
                                       columns=pd.DataFrame.from_dict(self.holdingPnlDict, orient='index').columns)

    # ----------------------------------------------------------------------
    def addData(self, key, dataframe, filename=None):
        """添加数据
        key: STRING ,the key of the data in datadict.
        dataframe: dataframe, the processed data.
        filename: STRING ,the name of the data file exist in the path,
        Note that the postfix 'h5' is  necessary here.

        """
        if filename:
            self.dataDict[key] = ff.readh5data(self.path, filename)
        else:
            self.dataDict[key] = dataframe


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
        denominator = pd.Series(self.totalMarketValueDict).reindex(self.backTestingDateList).fillna(method='ffill').dropna()
        return (nominator[denominator.index] / denominator).mean()

    # ----------------------------------------------------------------------
    # 平仓和开仓写成两个方法，使用时先撮合平仓单再撮合开仓单
    def crossAllOrder(self, date):
        """给定日期进行撮合及pnl/availableCash/marketValue的更新, 先平仓再反手"""
        try:
            openPrice = self.dataDict['adjOpen'].loc[date]
            closePrice = self.dataDict['adjClose'].loc[date]
            trueVolume = self.dataDict['volume'].loc[date]
        except KeyError:
            print date
            print 'No available data! Plz feed data!'

        #self.tradingList = []
        self.tradingPnlToday = {}  # 交易收益
        self.tradingAmountToday = 0  # 当日成交额
        self.tradingCommisionToday = 0  # 当日总手续费
        for order in self.allOrdersDict[date]:
            if not (np.isnan(openPrice[order.stkID]) or np.isnan(trueVolume[order.stkID])):
                if np.floor(0.05 * trueVolume[order.stkID]) <= 1:  # 股票成交量的5%小于1则不交易
                    continue
                crossPrice = openPrice[order.stkID]  # 最优价格
                dealPrice = crossPrice    # 成交价格
                dealVolume = min(order.volume, np.floor(0.05 * trueVolume[order.stkID]) * 100)  # 最多成交当天交易量的5%
                trade = OrderData(order.datetime, order.stkID, dealVolume, crossPrice, order.direction, order.offset)
                if trade.offset == Cover:
                    self.crossCoverOrder(trade)
                else:
                    self.crossOpenOrder(trade)
            else:
                continue

        # 计算当日交易完后的总市值（收盘时）
        #self.totalMarketValueDict[date] = self.calTotalMarketValue(closePrice)

        # 计算当日交易完后的可用资金
        self.availableCashDict[date] = self.availableCashNow

        # 添加当日总交易额至字典
        self.totalTradingAmountDict[date] = self.tradingAmountToday

        # 添加当日总交易手续费至字典
        self.totalCommisionDict[date] = self.tradingCommisionToday

        # 当日交易Pnl
        self.tradingPnlDict[date] = self.tradingPnlToday

        # 保存当日持仓字典
        #copyCP = copy.deepcopy(self.currentPositionDict)
        #self._allCurrentPositionDict[date] = copyCP



    # ----------------------------------------------------------------------
    def crossOpenOrder(self, trade):
        """撮合开仓单，注意要在平仓单之后撮合，开仓支付保证金"""
       # trade = OrderData(order.datetime, order.stkID, order.volume, order.price, order.direction, order.offset)
        #if order.offset != Open:  # 如果不是开仓则直接跳过此次交易
         #   return
        trade.commisionCost = self.marginRatio[trade.stkID] * self.multiplier[trade.stkID] * \
                              self.commisionRate * trade.volume * trade.price  # 手续费
        self.tradingAmountToday += trade.volume * trade.price
        self.tradingCommisionToday += trade.commisionCost
        stkNetcash = self.marginRatio[trade.stkID] * self.multiplier[trade.stkID] * self.tickSize[trade.stkID] * \
                     trade.volume * trade.price * (1 + self.commisionRate)  # 支付出去的保证金 + 手续费
        if stkNetcash > self.availableCashNow:  # 账户剩余的钱不够支付保证金的话则无法开仓
            return
        self.availableCashNow -= stkNetcash  # 从账户里扣除相应的保证金
        holding = holdingData(0.1, 0.1, 1)
        if trade.stkID in self.currentPositionDict.keys():
            if trade.direction * self.currentPositionDict[trade.stkID].direction < 0:
                trade.commisionCost = 0
                print "Invalid Open Order!Please Cover ur current position before open new contracts!"
                return

            holding.volume = self.currentPositionDict[trade.stkID].volume + trade.volume  # 新的持仓量
            # 更新持仓成本（注意持仓成本这里是有正负的）
            holding.averageCost = (self.currentPositionDict[trade.stkID].averageCost * self.currentPositionDict[
                trade.stkID].volume +
                trade.commisionCost + trade.direction * trade.price * trade.volume) / holding.volume
            #holding.direction = trade.direction
            # print 'average cost:', holding.averageCost

        else:  # 若无持仓则直接开仓
            holding.volume = trade.volume
            holding.averageCost = (trade.commisionCost +
                                   trade.direction * trade.price * trade.volume) / holding.volume
        holding.direction = trade.direction
        self.currentPositionDict[trade.stkID] = holding  # 更新持仓信息
        self.tradingCommisionToday += trade.commisionCost  # 交易手续费累加

    # ----------------------------------------------------------------------
    def crossCoverOrder(self, trade):
        """撮合平仓单，注意应保证队列中平仓单中的优先位置，平仓获取之前支付的保证金
        To-dos:添加平仓指令超出现在持有仓位的话自动反手的功能"""
        #trade = OrderData(order.datetime, order.stkID, order.volume, order.price, order.direction, order.offset)
        #if trade.offset != Cover:  # 如果不是开仓则直接跳过此次交易
         #   return
        if trade.stkID not in self.currentPositionDict.keys():  # 无持仓的话则跳过交易
            print "Invalid Cover Order! No available asset to cover!"
            return
        if trade.direction * self.currentPositionDict[trade.stkID].direction > 0:  # 先判断买卖方向是否和持仓方向相同
            print "Invalid Cover Order！Should use Open order instead of cover order."
            return
        currentHolding = self.currentPositionDict[trade.stkID]
        holding = holdingData(0.1, 0.1, 1)
        trade.volume = min(trade.volume, currentHolding.volume)
        stkNetcash = self.marginRatio[trade.stkID] * self.multiplier[trade.stkID] * self.tickSize[trade.stkID] *\
                     trade.volume * trade.price * (1 - self.commisionRate)  # 支付出去的保证金 + 手续费
        self.availableCashNow += stkNetcash  # 从账户里拿回相应的保证金
        holdingVolume = currentHolding.volume - trade.volume
        trade.commisionCost = self.marginRatio[trade.stkID] * self.multiplier[trade.stkID] * \
                              self.commisionRate * trade.volume * trade.price
        if holdingVolume <= 0:  # 完全平仓的情况
            #trade.commisionCost = self.commisionRate * currentHolding.volume * trade.price
            tradingAmount = trade.price * currentHolding.volume
            thistradepnl = self.multiplier[trade.stkID] * self.tickSize[trade.stkID] * \
                           (trade.direction * trade.price - currentHolding.averageCost) \
                           * trade.volume - trade.commisionCost
            self.tradingPnlToday[trade.stkID] = thistradepnl
            del self.currentPositionDict[trade.stkID]  # 删除持仓信息

        else:
            holding.volume = holdingVolume
            tradingAmount = trade.price * trade.volume
            holding.averageCost = (self.currentHolding.averageCost * self.currentHolding.volume +\
                                   trade.commisionCost + trade.direction * trade.price * trade.volume) / holding.volume
            holding.direction = currentHolding.direction  # 方向和之前的持仓方向一样
            self.currentPositionDict[trade.stkID] = holding  # 更新持仓信息
        self.tradingCommisionToday += trade.commisionCost  # 交易手续费累加
        self.tradingAmountToday += tradingAmount


    # ----------------------------------------------------------------------
    def updateHoldingInfo(self, date):
        """每日收盘前更新总市值总持仓收益以及持仓字典"""
        holdingPnl = {}
        closePrice = self.dataDict['adjClose'].loc[date]
        self.totalMarketValueDict[date] = self.calTotalMarketValue(closePrice)  # 更新总市值

        for stkID, holding in self.currentPositionDict.iteritems():
            holdingPnl[stkID] = self.multiplier[stkID] * self.tickSize[stkID] * \
                                (holding.direction * closePrice.loc[stkID] - holding.averageCost) * holding.volume
        # 保存当日持仓Pnl至持仓Pnl信息字典
        self.holdingPnlDict[date] = holdingPnl

        # 保存当日持仓字典
        copyCP = copy.deepcopy(self.currentPositionDict)
        self._allCurrentPositionDict[date] = copyCP

    # ----------------------------------------------------------------------
    def output(self, content):
        """输出内容"""
        print str(datetime.now()) + "\t" + content

    # ----------------------------------------------------------------------
    def showBackTestingResult(self):
        """计算各指标，并作图"""

        totalcommision = pd.Series(self.totalCommisionDict).sum()
        turnoverrate = self.calTurnOverRatio()
        #winrate = pd.Series(self.winRateCal)
        #winrate = winrate[winrate > 0].shape[0] / winrate .shape[0]
        totaldf = self.calTotalAsset()  # 计算总资产
        networth = totaldf.sum(axis=1) / totaldf.sum(axis=1).iloc[0]
        annualizedRet = (networth.iloc[-1]) ** (250 / networth.shape[0]) - 1
        annualizedvol = np.sqrt(250) * networth.pct_change().iloc[1:].std()
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
        #self.output(u'胜率: \t%s' % formatNumber(winrate))
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




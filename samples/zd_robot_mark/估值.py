from backtrader.feeds import PandasData
import backtrader as bt
import matplotlib.pyplot as plt
import numpy as np

from zvt.domain import *
from zvt.api import *
import pandas as pd
plt.rcParams['font.sans-serif'] = ['SimSun']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

trade_col = {
    'ref': '交易的唯一标识符',
    'dir': '交易方向',
    'datein': '交易打开日期时间',
    'pricein': '进入交易的价格',
    'dateout': '交易关闭的日期',
    'priceout': '退出交易的价格',
    'chng%': '退出价格相对进入价格增长百分比',
    'pnl%': '交易的利润或损失占交易关闭时账户总值的百分比',
    'size': '交易中的最大成交数量',
    'value': '交易中的最大成交价值',
    'cumpnl': '累积利润或损失',
    'nbars': '交易持续的bar数',
    'mfe%': '最大有利变动幅度',
    'mae%': '最大不利变动幅度',
    'dataname': '证券代码',
    'cucommission': '累计佣金',
}


class PandasDataExtend(PandasData):
    # 增加线
    lines = ('pe_ttm_分位', 'peg', 'pcf_分位', 'ps_ttm_分位', 'pb_分位')
    params = (
        ('pe_ttm_分位', -6),
        ('peg', -5),
        ('pcf_分位', -4),
        ('ps_ttm_分位', -3),
        ('pb_分位', -2),)


class Trade_list(bt.Analyzer):
    """
    定义新分析者
    """

    def get_analysis(self):

        return self.trades

    def __init__(self):

        self.trades = []
        self.cumprofit = 0.0

    def notify_trade(self, trade):

        if trade.isclosed:

            brokervalue = self.strategy.broker.getvalue()

            dir = 'short'
            if trade.history[0].event.size > 0: dir = 'long'

            pricein = trade.history[len(trade.history) - 1].status.price
            priceout = trade.history[len(trade.history) - 1].event.price
            datein = bt.num2date(trade.history[0].status.dt)
            dateout = bt.num2date(trade.history[len(trade.history) - 1].status.dt)
            if trade.data._timeframe >= bt.TimeFrame.Days:
                datein = datein.date()
                dateout = dateout.date()

            pcntchange = 100 * priceout / pricein - 100
            pnl = trade.history[len(trade.history) - 1].status.pnlcomm
            pnlpcnt = 100 * pnl / brokervalue
            barlen = trade.history[len(trade.history) - 1].status.barlen
            pbar = pnl / barlen
            self.cumprofit += pnl

            size = value = 0.0
            for record in trade.history:
                if abs(size) < abs(record.status.size):
                    size = record.status.size
                    value = record.status.value
            comm = sum([i.event.commission for i in trade.history])  # 买卖手续费的和
            highest_in_trade = max(trade.data.high.get(ago=0, size=barlen + 1))
            lowest_in_trade = min(trade.data.low.get(ago=0, size=barlen + 1))
            hp = 100 * (highest_in_trade - pricein) / pricein
            lp = 100 * (lowest_in_trade - pricein) / pricein
            if dir == 'long':
                mfe = hp
                mae = lp
            if dir == 'short':
                mfe = -lp
                mae = -hp

            self.trades.append({'ref': trade.ref, 'ticker': trade.data._name, 'dir': dir,
                                'datein': datein, 'pricein': pricein, 'dateout': dateout, 'priceout': priceout,
                                'chng%': round(pcntchange, 2), 'pnl': pnl, 'pnl%': round(pnlpcnt, 2),
                                'size': size, 'value': value, 'cumpnl': self.cumprofit,
                                'nbars': barlen, 'pnl/bar': round(pbar, 2),
                                'mfe%': round(mfe, 2), 'mae%': round(mae, 2),
                                'dataname': trade.getdataname(),
                                'cucommission': comm,
                                })

class stampDutyCommissionScheme(bt.CommInfoBase):
    '''
    本佣金模式下，买入股票仅支付佣金，卖出股票支付佣金和印花税.
    '''
    params = (
        ('stamp_duty', 0.005),  # 印花税率
        ('commission', 0.001),  # 佣金率
        ('stocklike', True),
        ('commtype', bt.CommInfoBase.COMM_PERC),
    )

    def _getcommission(self, size, price, pseudoexec):
        '''
        If size is greater than 0, this indicates a long / buying of shares.
        If size is less than 0, it idicates a short / selling of shares.
        '''
        # ('stamp_duty', 0.005),  # 印花税率
        # ('commission', 0.001),  # 佣金率
        if size > 0:  # 买入，不考虑印花税
            commission_val = size * price * self.p.commission * 100
            if commission_val < 5:
                return 5
            return commission_val
        elif size < 0:  # 卖出，考虑印花税
            commission_val = - size * price * (self.p.stamp_duty + self.p.commission * 100)
            if commission_val < 5:
                return 5
            return commission_val
        else:
            return 0  # just in case for some reason the size is 0.


def get_index_data(start='2015-02-02', end='2015-05-31'):
    df = get_kdata(entity_id="index_sh_000001",
                   start_timestamp=start, end_timestamp=end)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.rename(columns={"timestamp": "date",
                            })
    df.reset_index(drop=False, inplace=True)
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    df['openinterest'] = 0
    df = df[['open', 'high', 'low', 'close', 'volume', 'openinterest']]
    return df


def get_data_back(code, start='2021-02-02', end='2021-02-25'):
    df = get_kdata(entity_id=china_stock_code_to_id(code),start_timestamp=start, end_timestamp=end, adjust_type='hfq')

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.rename(columns={"timestamp": "date",
                            })

    df.reset_index(drop=False, inplace=True)
    # df['date'] = df['date'].apply(lambda x:x.strftime("%Y-%m-%d"))
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    df['openinterest'] = 0
    df = df[['open', 'high', 'low', 'close', 'volume', 'openinterest']]
    return df


class MyStrategy(bt.Strategy):
    """
    """
    # 策略参数
    params = dict(
        rebal_monthday=[1],  # 每月1日执行再平衡
        printlog=False,
        max_hold_num=10,
        values_num=10,
    )

    def __init__(self,
                 sell_signal=pd.DataFrame(),
                 buy_signal=pd.DataFrame(),
                 timing_signal=pd.DataFrame()):
        self.log_file = open('C:/Users/32771/Desktopposition_log.txt', 'w')  # 用于输出仓位信息
        # self.log_file_dataframe = pd.ExcelWriter(f'C:/Users/32771/Desktop滚动-{str(self.p.values_num)}-每日持仓.xlsx')  # 用于输出仓位信息
        self.log_file_dataframe = pd.DataFrame()  # 用于输出仓位信息

        self.sell_signal = sell_signal
        self.buy_signal = buy_signal
        self.timing_signal = timing_signal
        # 上次交易股票的列表
        self.lastRanks = []
        # 0号是指数，不进入选股池，从1号往后进入股票池
        self.stocks = self.datas[1:]
        # 记录以往订单，在再平衡日要全部取消未成交的订单
        self.order_list = []
        # 最大仓位，预留5%的仓位
        self.timing_data = 0.95
        self.end_date = self.data0.datetime.date(-1)
        self.start_date = self.data0.datetime.date(0)

        # 保存交易计划
        self.trade_plan = pd.DataFrame(columns=['股价', '股价变动幅度', '买入比例', '计划执行次数',
                                                '证券代码', '交易市场', '交易档位', '计划创建时间',
                                                '股份',
                                                ])
        self.trade_plan.set_index(['证券代码', '交易市场', '交易档位'], inplace=True, drop=True)
        # 是否为调仓日
        self.warehouse_date = False
        # 定时器
        self.add_timer(
            when=bt.Timer.SESSION_START,
            monthdays=self.p.rebal_monthday,  # 每月1号触发再平衡
            monthcarry=True,  # 若再平衡日不是交易日，则顺延触发notify_timer
        )

    def prenext(self):
        # 即使有些股票尚无当前数据，也跳转next执行
        self.next()

    def values_trade_plane(self, data):

        trade_plan = False
        if data._name.split('_')[-1] == '电子I' \
                and (data.peg < 1 or data.ps_ttm_分位 < self.p.values_num or data.pb_分位 < self.p.values_num):
            trade_plan = True
        elif data._name.split('_')[-1] in ['家用电器I','电气设备I'] \
                and (data.peg < 1 or data.pcf_分位 < self.p.values_num or data.ps_ttm_分位 < self.p.values_num):
            trade_plan = True
        elif data._name.split('_')[-1] == '食品饮料I' \
                and (data.peg < 1 or data.pb_分位 < self.p.values_num):
            trade_plan = True
        elif data._name.split('_')[-1] == '医药生物I' \
                and (data.peg < 1 or data.pb_分位 < self.p.values_num):
            trade_plan = True
        elif data._name.split('_')[-1] == '交通运输I' \
                and (data.pcf_分位 < self.p.values_num or data.pb_分位 < self.p.values_num):
            trade_plan = True
        elif data._name.split('_')[-1] == '化工I' \
                and (data.peg < 1 or data.pcf_分位 < self.p.values_num or data.ps_ttm_分位 < self.p.values_num):
            trade_plan = True
        elif data._name.split('_')[-1] == '房地产I' \
                and (data.peg < 1 or data.pcf_分位 < self.p.values_num or data.ps_ttm_分位 < self.p.values_num):
            trade_plan = True
        # 信息服务
        elif data._name.split('_')[-1] == '计算机I' \
                and (
                data.pb_分位 < self.p.values_num or data.pe_ttm_分位 < self.p.values_num or data.ps_ttm_分位 < self.p.values_num):
            trade_plan = True
        elif data._name.split('_')[-1] == '轻工制造I' \
                and (data.peg < 1 or data.ps_ttm_分位 < self.p.values_num):
            trade_plan = True
        elif data._name.split('_')[-1] == '采掘I' \
                and (data.pe_ttm_分位 < self.p.values_num or data.pcf_分位 < self.p.values_num):
            trade_plan = True
        return trade_plan

    def next(self):
        # 倒数第二个交易日，清仓
        if self.end_date == self.data0.datetime.date(0):
            # 当前持有仓位的股票列表
            posdata = [d for d, pos in self.getpositions().items() if pos]
            for d in (d for d in posdata):
                self.order_target_percent(d, target=0.0)  # 清仓
        # 调仓日 不执行交易计划，避免重复交易
        if not self.warehouse_date:
            # 执行交易计划
            # 下单
            for data in self.lastRanks:
                buy_position = plan_detil = None
                # 已有仓位，不买入
                if self.broker.getvalue([data]):
                    continue
                # 必须有应有交易计划,且满足估值要求
                if len(data) < data.buflen():
                    trade_plan = self.values_trade_plane(data)
                    if trade_plan:
                        trade_plan_data = self.trade_plan[self.trade_plan.index.get_level_values('证券代码') == data._name]

                        plan_detil = '买入点'
                        buy_position = \
                            trade_plan_data[
                                trade_plan_data.index.get_level_values('交易档位') == plan_detil].买入比例.values[0]

                        if buy_position:
                            self.trade_plan.loc[(data._name, 'XSHG', plan_detil), '计划执行次数'] += 1
                            # 按照次日开盘价计算下单量,下单量是100的整数倍
                            size = int(
                                self.trade_plan.loc[(data._name, 'XSHG', plan_detil), '计划可用资金'] / 100 / data.open[
                                    1]) * 100 * buy_position
                            self.log(f'*** {self.data0.datetime.date(0)},交易计划执行,{data._name}, 买入{size}股')
                            self.buy(data=data, size=size)
        self.warehouse_date = False
        # # 打印仓位信息
        self.trading_position()

    def notify_timer(self, timer, when, *args, **kwargs):
        self.rebalance_portfolio()  # 执行再平衡

    def rebalance_portfolio(self):
        # 从指数取得当前日期
        self.currDate = self.data0.datetime.date(0)
        # 如果是指数的最后一本bar，则退出，防止取下一日开盘价越界错
        if len(self.datas[0]) == self.data0.buflen():
            return
        # 取消以往所下订单（已成交的不会起作用）
        for o in self.order_list:
            self.cancel(o)
        self.order_list = []  # 重置订单列表
        # 股票池
        buy_signale_curr = self.buy_signal.query("timestamp <= @self.currDate")
        long_list = buy_signale_curr.loc[buy_signale_curr.index.max()].CODES.tolist()
        long_list = [i for i in long_list if 'nan' not in i]
        # 最终
        self.ranks = [d for d in self.stocks if
                      len(d) > 0  # 重要，到今日至少要有一根实际bar
                      and d._name in long_list
                      # 今日未停牌 (若去掉此句，则今日停牌的也可能进入，并下订单，次日若复牌，则次日可能成交）（假设原始数据中已删除无交易的记录)
                      ]
        # 同时出现在 两次入选股票但是没有持仓，需要重置交易计划
        ranks_duplicate = set(d for d in self.ranks if
                              d in self.lastRanks
                              and self.broker.getvalue([d]) <= 0)

        self.long_ranks = set(self.ranks) - set(self.lastRanks)
        # 无新股票选中，且 无重复入选股票未持仓 则退出 ,否则 合并为本次做多信号
        if len(self.long_ranks) == 0 and len(ranks_duplicate) == 0:
            return
        else:
            self.long_ranks = self.long_ranks | ranks_duplicate
        # 以往买入的标的，本次不在标的中，则先平仓
        data_toclose = set(self.lastRanks) - set(self.ranks)
        for d in data_toclose:
            o = self.close(data=d)
            self.order_list.append(o)  # 记录订单

        # 得到当前的账户价值
        total_value = self.broker.getvalue()
        # 以往标的和本次标的重复的持仓市值
        last_ranks_values = sum(
            [self.broker.getvalue([i])
             for i in set(self.lastRanks) & set(self.ranks) if
             self.broker.getvalue([i])])
        # 总资产 - 不清仓的市值 获得剩余现金进行仓位分配
        self.p_value = (total_value - last_ranks_values) * self.timing_data / len(self.long_ranks)
        # 下单
        for data in self.long_ranks:
            # 查询该股的交易计划，如果有，说明在上个周期没有执行，应该删除上次的计划
            trade_plan_data = self.trade_plan[self.trade_plan.index.get_level_values('证券代码') == data._name]
            if not trade_plan_data.empty:
                self.trade_plan = self.trade_plan.drop((data._name, 'XSHG'))

            # 按照次日开盘价计算下单量,下单量是100的整数倍
            if len(data) < data.buflen():
                # 判断是否满足估值条件
                trade_plan = self.values_trade_plane(data)
                # 满足就买入，不满足就创建交易计划，后续执行
                if trade_plan:
                    # 必然为初次买入
                    size = int(self.p_value / 100 / data.open[1]) * 100
                    self.buy(data=data, size=size)
                    # 成交后才生成交易计划，在notify_order中生成
                else:
                    # 初次买入
                    self.trade_plan.loc[(data._name, 'XSHG', '买入点'), '股价变动幅度'] = '0%'
                    self.trade_plan.loc[(data._name, 'XSHG', '买入点'), '买入比例'] = 1
                    self.trade_plan.loc[(data._name, 'XSHG', '买入点'), '计划执行次数'] = int(1)
                    self.trade_plan.loc[
                        (data._name, 'XSHG', '买入点'), '计划可用资金'] = self.p_value

                    self.trade_plan.loc[(data._name, 'XSHG'), '计划创建时间'] = data.datetime.date(0)
        self.lastRanks = self.ranks  # 跟踪上次买入的标的
        self.warehouse_date = True
        # # 打印仓位信息
        self.trading_position()

    def trading_position(self):
        """
        记录仓位信息，stop时保存到指定路径
        Returns:

        """
        log_file_dataframe = pd.DataFrame()
        log_date = self.data0.datetime.date(0).strftime("%Y%m%d")
        if log_date not in log_file_dataframe.index:
            log_file_dataframe = log_file_dataframe.append(pd.DataFrame({
                '总资产': self.broker.getvalue(),
                '总现金': self.broker.getcash(),
            }, index=[log_date]))

        for i, d in enumerate(self.datas):
            if d._name == '000001':
                continue
            pos = self.getposition(d)
            if pos:
                # date_index = pos.datetime.strftime("%Y%m%d")
                log_file_dataframe = log_file_dataframe.append(pd.DataFrame({'股票代码': d._name,
                                                                             '持仓': pos.size,
                                                                             '成本价': pos.price,
                                                                             '当前价': pos.adjbase,
                                                                             '盈亏': pos.size * (
                                                                                     pos.adjbase - pos.price)},
                                                                            index=[log_date]))

        if not log_file_dataframe.empty:
            # log_file_dataframe.to_excel(self.log_file_dataframe, sheet_name=f'{date_index}')
            self.log_file_dataframe = self.log_file_dataframe.append(log_file_dataframe)

    def stop(self):
        self.log_file.close()
        self.log_file_dataframe.to_excel(f'C:/Users/32771/Desktop/估值-{str(self.p.values_num)}-每日持仓.xlsx')


    def log(self, txt, dt=None, doprint=False):
        if self.params.printlog or doprint:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()},{txt}')

    # 记录交易执行情况（可省略，默认不输出结果）
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # 订单状态 submitted/accepted，无动作
            return

        # 订单完成
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log('买单执行,%s, %.2f, %i' % (order.data._name,
                                                order.executed.price, order.executed.size))
                if not self.trade_plan[self.trade_plan.index.get_level_values('证券代码') == order.data._name].empty:
                    # 买完应该删除计划
                    self.trade_plan = self.trade_plan.drop((order.data._name, 'XSHG'))

            elif order.issell():
                self.log('卖单执行, %s, %.2f, %i' % (order.data._name,
                                                 order.executed.price, order.executed.size))
        else:
            self.log('订单作废 %s, %s, isbuy=%i, size %i, open price %.2f' %
                     (order.data._name, order.getstatusname(), order.isbuy(), order.created.size, order.data.open[0]))

    # 记录交易收益情况（可省略，默认不输出结果）
    def notify_trade(self, trade):
        if trade.isclosed:
            print('毛收益 %0.2f, 扣佣后收益 % 0.2f, 佣金 %.2f, 市值 %.2f, 现金 %.2f' %
                  (trade.pnl, trade.pnlcomm, trade.commission, self.broker.getvalue(), self.broker.getcash()))


def test_start():
    def apply_code(codes):
        return codes[:6]

    import time
    from datetime import date
    back_date = ['20100101']
    # back_date = ['20190101']
    for ye in back_date:
        strategy_name = f'{ye}10'
        start = f'{ye}'
        end = '20210516'
        res_df = []

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        code_dict = {}
        buy_signal = pd.read_csv("C:/Users/32771/Documents/dev data/buy_signal.csv")
        buy_signal['timestamp'] = pd.to_datetime(buy_signal['timestamp'])
        buy_signal['股票代码']  = buy_signal['股票代码'].apply(lambda x:str(x).zfill(6))
        for stock_data in buy_signal[['股票代码', 'timestamp', '股票名称', '股票池']].to_dict('records'):
            stock_n = stock_data['股票代码'] + '_' + stock_data['股票名称']
            if stock_n not in code_dict:
                code_dict.update({stock_n: stock_data['timestamp']})
            else:
                if code_dict[stock_n] >= stock_data['timestamp']:
                    code_dict.update({stock_n: stock_data['timestamp']})
        buy_signal.set_index(['timestamp'],drop=True,inplace=True)
        code_dict.update({'000001_上证指数_指数': pd.to_datetime(start)})
        # for vlnum in [100, 80, 50, 60, 40, 20]:
        for vlnum in [50,60]:
            cerebro = bt.Cerebro(tradehistory=True)
            for name in sorted(code_dict):
                # data = save_csv(name)
                data = pd.read_csv(f"C:/Users/32771/Documents/dev data/0610/datachoices_滚动/{name[:6]}.csv")
                data['date'] = pd.to_datetime(data['date'])
                data.set_index(['date'], drop=True, inplace=True)
                data = data[['open', 'high', 'low', 'close', 'volume',
                             'pe_ttm_分位', 'peg', 'pcf_分位', 'ps_ttm_分位', 'pb_分位',
                             'openinterest']].query("date >= @start")
                if name != '000001_上证指数_指数':
                    name = buy_signal[buy_signal.股票代码 == name[:6]].CODES[0]
                feed = PandasDataExtend(
                    dataname=data,
                    open=0,  # 开盘价所在列
                    high=1,  # 最高价所在列
                    low=2,  # 最低价所在列
                    close=3,  # 收盘价价所在列
                    volume=4,  # 成交量所在列

                    pe_ttm_分位=-6,
                    peg=-5,
                    pcf_分位=-4,
                    ps_ttm_分位=-3,
                    pb_分位=-2,
                    openinterest=-1,  # 无未平仓量列.(openinterest是期货交易使用的)
                    fromdate=pd.to_datetime(data.index[0].strftime("%Y%m%d")),  # 起始日
                    todate=pd.to_datetime(data.index[-1].strftime("%Y%m%d"))  # 结束日
                )

                cerebro.adddata(feed, name=name)
                del data, feed

            # 回测设置
            startcash = 20000000
            cerebro.broker.setcash(startcash)
            # 防止下单时现金不够被拒绝。只在执行时检查现金够不够。
            cerebro.broker.set_checksubmit(False)
            # 买入股票仅支付佣金，卖出股票支付佣金和印花税.
            comminfo = stampDutyCommissionScheme(stamp_duty=0.001, commission=0.001)
            cerebro.broker.addcommissioninfo(comminfo)
            # 固定滑点
            cerebro.broker.set_slippage_fixed(0.05)

            # add analyzers
            cerebro.addanalyzer(Trade_list, _name='trade_list')
            cerebro.broker.set_checksubmit(False)

            # 添加策略
            cerebro.addstrategy(MyStrategy, values_num=vlnum, buy_signal=buy_signal, printlog=True)
            df00, df0, df1, df2, df3, df4, trade_list_df = bt.out_result(cerebro)
            # cerebro.run()
            # b=Bokeh(style='bar',tabs='multi',sched=Tradimo()) #传统白底，多页
            # cerebro.plot(b)
            df0[f'{vlnum}%分位低估'] = df0[f'total_value']
            df0[[f'{vlnum}%分位低估']].to_excel(f'C:/Users/32771/Desktop/{strategy_name}-{vlnum}-估值表现.xlsx')
            res_df.append(df0[[f'{vlnum}%分位低估']])
            trade_list_df.rename(columns=trade_col, inplace=True)
            trade_list_df.to_excel(f'C:/Users/32771/Desktop/{strategy_name}-{vlnum}-交易详情.xlsx')
            del df0, vlnum



def save_csv(name):
    data = pd.read_csv(f"C:/Users/32771/Documents/dev data/0604/行情/{name[:6]}.csv")

    data['date'] = pd.to_datetime(data['date'])
    # 000022 - 估值, 没数据
    if name[:6] != '000001':
        try:
            data_values = pd.read_excel(f"C:/Users/32771/Documents/dev data/0610/滚动/"
                                        f"{name[:6]}-{name.split('_')[-1].replace('*', '')}-估值.xlsx")
            data_values['date'] = pd.to_datetime(data_values['date'])
            data_values['peg'] = data_values['PEG']
        except:
            print(f"{name[:6]}-估值,没数据")
            return
        data_values['date'] = pd.to_datetime(data_values['date'])
        name = name + '_' + data_values['行业分类'].iloc[0]
    else:
        data_values = pd.DataFrame(columns=['pe_ttm_分位', 'peg', 'pcf_分位', 'pb_分位', 'ps_ttm_分位', '行业分类'])
        data_values['date'] = data['date']

    data = pd.merge(data_values, data, on=['date'])
    data = data[['date', 'open', 'high', 'low', 'close', 'volume',
                 'pe_ttm_分位', 'peg', 'pcf_分位', 'ps_ttm_分位', 'pb_分位',
                 'openinterest']]
    data.to_csv(f'C:/Users/32771/Documents/dev data/0610/datachoices_滚动/{name[:6]}.csv')
    data.set_index(['date'], drop=True, inplace=True)
    return data

test_start()
import time
from datetime import timedelta
from backtrader.feeds import PandasData
import backtrader as bt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# from EmQuantAPI import *

# loginResult = c.start("ForceLogin=1")
from zvt.api import *
from zvt.domain import *

plt.rcParams["font.sans-serif"] = ["SimSun"]  # 用来正常显示中文标签
plt.rcParams["axes.unicode_minus"] = False  # 解决保存图像是负号'-'显示为方块的问题

trade_col = {
    "ref": "交易的唯一标识符",
    "dir": "交易方向",
    "datein": "交易打开日期时间",
    "pricein": "进入交易的价格",
    "dateout": "交易关闭的日期",
    "priceout": "退出交易的价格",
    "chng%": "退出价格相对进入价格增长百分比",
    "pnl%": "交易的利润或损失占交易关闭时账户总值的百分比",
    "size": "交易中的最大成交数量",
    "value": "交易中的最大成交价值",
    "cumpnl": "累积利润或损失",
    "nbars": "交易持续的bar数",
    "mfe%": "最大有利变动幅度",
    "mae%": "最大不利变动幅度",
    "dataname": "证券代码",
    "cucommission": "累计佣金",
}


class PandasDataExtend(PandasData):
    # 增加线
    lines = ("pe_ttm_分位", "peg", "pcf_分位", "ps_ttm_分位", "pb_分位")
    params = (
        ("pe_ttm_分位", -6),
        ("peg", -5),
        ("pcf_分位", -4),
        ("ps_ttm_分位", -3),
        ("pb_分位", -2),
    )


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

            dir = "short"
            if trade.history[0].event.size > 0:
                dir = "long"

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
            # 持仓数  持仓市值
            size = value = 0.0
            # 买入数量   第一次买入数量  第二次买入数量   第三次买入数量
            buy_count = first_size = second_size = third_size = 0
            first_price = second_price = third_price = 0
            first_date = second_date = third_date = 0
            # 最大三次买入数量 之后 应等于 size
            for record in trade.history:

                if abs(size) < abs(record.status.size):
                    size = record.status.size
                    value = record.status.value
                if record.status.status == 1 and buy_count == 0:
                    # 第一次买入
                    first_size = record.event.size
                    first_price = record.event.price
                    first_date = bt.num2date(record.status.dt)
                    buy_count += 1
                elif record.status.status == 1 and buy_count == 1:
                    # 第二次买入
                    second_size = record.event.size
                    second_price = record.event.price
                    second_date = bt.num2date(record.status.dt)
                    buy_count += 1
                elif record.status.status == 1 and buy_count == 2:
                    # 第三次买入
                    third_size = record.event.size
                    third_price = record.event.price
                    third_date = bt.num2date(record.status.dt)
                    buy_count += 1
            comm = sum([i.event.commission for i in trade.history])  # 买卖手续费的和
            highest_in_trade = max(trade.data.high.get(ago=0, size=barlen + 1))
            lowest_in_trade = min(trade.data.low.get(ago=0, size=barlen + 1))
            hp = 100 * (highest_in_trade - pricein) / pricein
            lp = 100 * (lowest_in_trade - pricein) / pricein
            if dir == "long":
                mfe = hp
                mae = lp
            if dir == "short":
                mfe = -lp
                mae = -hp

            self.trades.append(
                {
                    "ref": trade.ref,
                    "ticker": trade.data._name,
                    "dir": dir,
                    "datein": datein,
                    "pricein": pricein,
                    "dateout": dateout,
                    "priceout": priceout,
                    "chng%": round(pcntchange, 2),
                    "pnl": pnl,
                    "pnl%": round(pnlpcnt, 2),
                    "size": size,
                    "value": value,
                    "cumpnl": self.cumprofit,
                    "nbars": barlen,
                    "pnl/bar": round(pbar, 2),
                    "mfe%": round(mfe, 2),
                    "mae%": round(mae, 2),
                    "dataname": trade.getdataname(),
                    "cucommission": comm,
                    "第一次买入数量": first_size,
                    "第一次买入价格": first_price,
                    "第一次买入时间": first_date,
                    "第二次买入数量": second_size,
                    "第二次买入价格": second_price,
                    "第二次买入时间": second_date,
                }
            )


class stampDutyCommissionScheme(bt.CommInfoBase):
    """
    本佣金模式下，买入股票仅支付佣金，卖出股票支付佣金和印花税.
    """

    params = (
        ("stamp_duty", 0.005),  # 印花税率
        ("commission", 0.001),  # 佣金率
        ("stocklike", True),
        ("commtype", bt.CommInfoBase.COMM_PERC),
    )

    def _getcommission(self, size, price, pseudoexec):
        """
        If size is greater than 0, this indicates a long / buying of shares.
        If size is less than 0, it idicates a short / selling of shares.
        """
        # ('stamp_duty', 0.005),  # 印花税率
        # ('commission', 0.001),  # 佣金率
        if size > 0:  # 买入，不考虑印花税
            commission_val = size * price * self.p.commission * 100
            if commission_val < 5:
                return 5
            return commission_val
        elif size < 0:  # 卖出，考虑印花税
            commission_val = (
                -size * price * (self.p.stamp_duty + self.p.commission * 100)
            )
            if commission_val < 5:
                return 5
            return commission_val
        else:
            return 0  # just in case for some reason the size is 0.


def get_index_data(start="2015-02-02", end="2015-05-31"):
    df = get_kdata(
        entity_id="index_sh_000001", start_timestamp=start, end_timestamp=end
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.rename(
        columns={
            "timestamp": "date",
        }
    )
    df.reset_index(drop=False, inplace=True)
    df.set_index("date", inplace=True)
    df.sort_index(inplace=True)
    df["openinterest"] = 0
    df = df[["open", "high", "low", "close", "volume", "openinterest"]]
    return df


def get_data_back(code, start="2021-02-02", end="2021-02-25"):
    df = get_kdata(
        entity_id=china_stock_code_to_id(code),
        start_timestamp=start,
        end_timestamp=end,
        adjust_type="hfq",
    )

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.rename(
        columns={
            "timestamp": "date",
        }
    )

    df.reset_index(drop=False, inplace=True)
    # df['date'] = df['date'].apply(lambda x:x.strftime("%Y-%m-%d"))
    df.set_index("date", inplace=True)
    df.sort_index(inplace=True)
    df["openinterest"] = 0
    df = df[["open", "high", "low", "close", "volume", "openinterest"]]
    return df


class MyStrategy(bt.Strategy):
    """ """

    # 策略参数
    params = dict(
        strategy_name="成长价值一号",
        rebal_monthday=[1],  # 每月1日执行再平衡
        printlog=False,
        max_hold_num=10,  # 最大持股数
        values_num=10,  # 估值分位数
        # 减持数据
        HolderTradingData=pd.DataFrame(
            columns=["holder_name", "change_pct", "code", "timestamp"], index=[0]
        ),
        # 质押比例数据
        EquityPledgeData=pd.DataFrame(
            columns=[
                "pledge_total_ratio",
                "start_date",
                "end_date",
                "code",
                "timestamp",
            ],
            index=[0],
        ),
        # 应收账款数据
        BalanceSheetData=pd.DataFrame(
            columns=["accounts_receivable", "report_date", "code", "timestamp"],
            index=[0],
        ),
        # 营业收入数据
        IncomeStatementData=pd.DataFrame(
            columns=["operating_income", "report_date", "code", "timestamp"], index=[0]
        ),
        risk_pledge_ratio=50,  # 风控，股票质押率
        risk_accounts_receivable_to_operating_income=50,  # 风控，应收账款比营业收入
        risk_holdtrading=5,  # 风控，过去一年董高监减持总量占总股本的比例
    )

    def __init__(
        self,
        sell_signal=pd.DataFrame(),
        buy_signal=pd.DataFrame(),
        timing_signal=pd.DataFrame(),
    ):
        self.log_file = open(
            f"C:/Users/32771/Desktop/回测/风控/{self.p.strategy_name}-position_log.txt", "w"
        )  # 用于输出仓位信息
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

        # 保存风控原因
        self.risk_reason = pd.DataFrame(
            columns=["证券代码", "交易市场", "风控原因", "计算时间"]
        ).set_index(["计算时间", "证券代码", "交易市场"], drop=True)

        # 保存交易计划
        self.trade_plan = pd.DataFrame(
            columns=[
                "股价",
                "股价变动幅度",
                "买入比例",
                "计划执行次数",
                "证券代码",
                "交易市场",
                "交易档位",
                "计划创建时间",
                "股份",
                "买入原因",
            ]
        )
        self.trade_plan.set_index(["证券代码", "交易市场", "交易档位"], inplace=True, drop=True)

        # 交易原因
        self.transaction_reason = pd.DataFrame()
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

    def risk_model(self, data):
        """
        风控模型
        Returns:
            True 为有风险
            False 为无风险
        """
        risk_code = data._name[:6]
        risk_data = False  # 先赋值为无风险，后续进行判断
        risk_date = self.data0.datetime.date(0).strftime("%Y%m%d")  # 风控计算日期
        risk_date_1 = (self.data0.datetime.date(0) - timedelta(days=365)).strftime(
            "%Y%m%d"
        )  # 风控计算日期

        # 质押比例
        EquityPledgeData = self.p.EquityPledgeData.query(
            f"code == '{data._name[:6]}' "
            f"and '{risk_date}' >= start_date "
            f"and '{risk_date}' <= end_date"
        ).pledge_total_ratio.sum()

        if EquityPledgeData > 50:
            self.risk_reason = self.risk_reason.append(
                pd.DataFrame(
                    {
                        "风控原因": f"股票质押率为{EquityPledgeData}%，超过{self.p.risk_pledge_ratio}%。",
                        "计算时间": risk_date,
                        "证券代码": data._name,
                        "交易市场": "XSHG",
                    },
                    index=[0],
                ).set_index(["计算时间", "证券代码", "交易市场"], drop=True)
            )
            risk_data = True

        # 收入中应收账款占比不能超过50
        # 应收账款
        BalanceSheetData = (
            self.p.BalanceSheetData.query(
                f"code == '{data._name[:6]}' and '{risk_date}' >= timestamp"
            )
            .sort_values("report_date", ascending=False)
            .iloc[:1]
        )
        # 营业收入
        IncomeStatementData = (
            self.p.IncomeStatementData.query(
                f"code == '{data._name[:6]}'  and '{risk_date}' >= timestamp"
            )
            .sort_values("report_date", ascending=False)
            .iloc[:1]
        )
        accounts_receivable_to_operating_income = (
            BalanceSheetData.accounts_receivable.values[0]
            / IncomeStatementData.operating_income.values[0]
        ) * 100
        # 判断两个数据是否为同一财报，否则提示
        if IncomeStatementData.report_date.values[0] != BalanceSheetData.report_date.values[0]:
            print(
                f"股票在：{data._name[:6]}，"
                f"在{risk_date}日"
                f"获取的{pd.to_datetime(BalanceSheetData.report_date.values[0]).strftime('%Y%m%d')}财务数据有问题，请检查！"
            )
        else:
            if (
                accounts_receivable_to_operating_income
                > self.p.risk_accounts_receivable_to_operating_income
            ):
                self.risk_reason = self.risk_reason.append(
                    pd.DataFrame(
                        {
                            "风控原因": f"收入中应收账款占比为{accounts_receivable_to_operating_income}%，超过{self.p.risk_accounts_receivable_to_operating_income}%。",
                            "计算时间": risk_date,
                            "证券代码": data._name,
                            "交易市场": "XSHG",
                        },
                        index=[0],
                    ).set_index(["计算时间", "证券代码", "交易市场"], drop=True)
                )
                risk_data = True

        HolderTradingData = self.p.HolderTradingData.query(
            f"code == '{data._name[:6]}'  and '{risk_date_1}' <= timestamp and '{risk_date}' >= timestamp"
        )

        if not HolderTradingData.empty:
            HolderTradingData_pct = HolderTradingData.groupby(
                "holder_name"
            ).change_pct.sum()
            for hold_name, hold_num in HolderTradingData_pct.to_dict().items():
                if hold_num > 5:
                    self.risk_reason = self.risk_reason.append(
                        pd.DataFrame(
                            {
                                "风控原因": f"过去一年中{hold_name}合计减持总量为总股本的{hold_num}%，超过{self.p.risk_holdtrading}%。",
                                "计算时间": risk_date,
                                "证券代码": data._name,
                                "交易市场": "XSHG",
                            },
                            index=[0],
                        ).set_index(["计算时间", "证券代码", "交易市场"], drop=True)
                    )
                    risk_data = True
        return risk_data

    #
    def values_trade_plane(self, data):
        """
        根据股票所处行业，判断其估值是否满足买点
        Args:
            data:

        Returns:

        """
        trade_plan = False
        if data._name.split("_")[-1] == "电子I" and (
            data.peg < 1
            or data.ps_ttm_分位 < self.p.values_num
            or data.pb_分位 < self.p.values_num
        ):
            trade_plan = True
        elif data._name.split("_")[-1] in ["家用电器I", "电气设备I"] and (
            data.peg < 1
            or data.pcf_分位 < self.p.values_num
            or data.ps_ttm_分位 < self.p.values_num
        ):
            trade_plan = True
        elif data._name.split("_")[-1] == "食品饮料I" and (
            data.peg < 1 or data.pb_分位 < self.p.values_num
        ):
            trade_plan = True
        elif data._name.split("_")[-1] == "医药生物I" and (
            data.peg < 1 or data.pb_分位 < self.p.values_num
        ):
            trade_plan = True
        elif data._name.split("_")[-1] == "交通运输I" and (
            data.pcf_分位 < self.p.values_num or data.pb_分位 < self.p.values_num
        ):
            trade_plan = True
        elif data._name.split("_")[-1] == "化工I" and (
            data.peg < 1
            or data.pcf_分位 < self.p.values_num
            or data.ps_ttm_分位 < self.p.values_num
        ):
            trade_plan = True
        elif data._name.split("_")[-1] == "房地产I" and (
            data.peg < 1
            or data.pcf_分位 < self.p.values_num
            or data.ps_ttm_分位 < self.p.values_num
        ):
            trade_plan = True
        # 信息服务
        elif data._name.split("_")[-1] == "计算机I" and (
            data.pb_分位 < self.p.values_num
            or data.pe_ttm_分位 < self.p.values_num
            or data.ps_ttm_分位 < self.p.values_num
        ):
            trade_plan = True
        elif data._name.split("_")[-1] == "轻工制造I" and (
            data.peg < 1 or data.ps_ttm_分位 < self.p.values_num
        ):
            trade_plan = True
        elif data._name.split("_")[-1] == "采掘I" and (
            data.pe_ttm_分位 < self.p.values_num or data.pcf_分位 < self.p.values_num
        ):
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
            for index, data in enumerate(self.lastRanks):
                buy_position = False
                trade_plan_data = self.trade_plan[
                    self.trade_plan.index.get_level_values("证券代码") == data._name
                    ]
                # 检查股票风控，触发即删除交易计划并清仓，不需要从lastRanks中删除，因为交易计划已删除，不会在该周期内再次买入
                if self.risk_model(data):
                    if not trade_plan_data.empty:
                        self.trade_plan = self.trade_plan.drop(
                            (data._name, "XSHG")
                        )  # 删除交易计划
                    if self.broker.getvalue([data]):
                        o = self.close(data=data)
                        self.order_list.append(o)  # 有持仓就清仓，记录订单
                    self.lastRanks.remove(data) # 从做多列表中删除避免重复记录

                if (
                    not trade_plan_data.empty
                    and self.broker.getvalue([data])
                    and trade_plan_data.计划执行次数[0] < 1
                ):
                    buy_price = trade_plan_data[
                        trade_plan_data.index.get_level_values("交易档位") == "买入点"
                    ].股价.values[0]
                    # 必须有应有交易计划,且满足估值要求
                    if len(data) < data.buflen():
                        # 已有仓位，且有交易计划 通过金字塔买入过，判断是否满足估值
                        trade_plan = self.values_trade_plane(data)
                        # 满足估值就买剩下的仓位
                        if trade_plan:
                            buy_position = True
                            self.trade_plan.loc[
                                (data._name, "XSHG", "买入点"), "买入原因"
                            ] = "估值"
                        # 不满足就判断金字塔条件
                        elif buy_price >= data.open[1]:
                            buy_position = True
                            self.trade_plan.loc[
                                (data._name, "XSHG", "买入点"), "买入原因"
                            ] = "金字塔"
                        if buy_position:
                            # 按照次日开盘价计算下单量,下单量是100的整数倍
                            size = (
                                (
                                    self.trade_plan.loc[
                                        (data._name, "XSHG", "买入点"), "计划可用资金"
                                    ]
                                    / data.open[1]
                                )
                                // 100
                            ) * 100
                            self.log(
                                f"*** {self.data0.datetime.date(0)},估值交易计划执行,{data._name}, 买入{size}股"
                            )
                            self.buy(data=data, size=size)
                            self.trade_plan.loc[
                                (data._name, "XSHG", "买入点"), "计划执行次数"
                            ] = int(1)

        self.warehouse_date = False
        # 打印仓位信息
        self.trading_position()

    def trading_position(self):
        """
        记录仓位信息，stop时保存到指定路径
        Returns:

        """
        log_file_dataframe = pd.DataFrame()
        log_date = self.data0.datetime.date(0).strftime("%Y%m%d")
        if log_date not in log_file_dataframe.index:
            log_file_dataframe = log_file_dataframe.append(
                pd.DataFrame(
                    {
                        "总资产": self.broker.getvalue(),
                        "总现金": self.broker.getcash(),
                    },
                    index=[log_date],
                )
            )

        for i, d in enumerate(self.datas):
            if d._name == "000001":
                continue
            pos = self.getposition(d)
            if pos:
                # date_index = pos.datetime.strftime("%Y%m%d")
                log_file_dataframe = log_file_dataframe.append(
                    pd.DataFrame(
                        {
                            "股票代码": d._name,
                            "持仓": pos.size,
                            "成本价": pos.price,
                            "当前价": pos.adjbase,
                            "盈亏": pos.size * (pos.adjbase - pos.price),
                        },
                        index=[log_date],
                    )
                )

        if not log_file_dataframe.empty:
            self.log_file_dataframe = self.log_file_dataframe.append(log_file_dataframe)

    def notify_timer(self, timer, when, *args, **kwargs):
        self.rebalance_portfolio()  # 执行再平衡

    def rebalance_portfolio(self):
        # 从指数取得当前日期
        self.currDate = self.data0.datetime.date(0)
        if self.currDate.month != 5 and self.lastRanks != []:
            # print(self.currDate, self.lastRanks)
            return
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
        long_list = [i for i in long_list if "nan" not in i]
        # 最终
        self.ranks = [
            d
            for d in self.stocks
            if len(d) > 0  # 重要，到今日至少要有一根实际bar
            and d._name in long_list
            and not self.risk_model(d)  # 不触发风控，否则不加入
        ]
        # 同时出现在 两次入选股票但是没有持仓，需要重置交易计划
        ranks_duplicate = set(
            d
            for d in self.ranks
            if d in self.lastRanks and self.broker.getvalue([d]) <= 0
        )

        self.long_ranks = set(self.ranks) - set(self.lastRanks)
        # 无新股票选中，且无重复入选股票未持仓 则退出 ,否则 合并为本次做多信号
        if len(self.long_ranks) == 0 and len(ranks_duplicate) == 0:
            return
        else:
            self.long_ranks = self.long_ranks | ranks_duplicate
        # 以往买入的标的，本次不在标的中，则先平仓
        data_toclose = set(self.lastRanks) - set(self.ranks)
        for d in data_toclose:
            # print('sell 平仓', d._name, self.getposition(d).size)
            o = self.close(data=d)
            self.order_list.append(o)  # 记录订单

        # 重置交易计划
        self.trade_plan = pd.DataFrame(
            columns=[
                "股价",
                "股价变动幅度",
                "买入比例",
                "计划执行次数",
                "证券代码",
                "交易市场",
                "交易档位",
                "计划创建时间",
                "股份",
                "买入原因",
            ]
        )
        self.trade_plan.set_index(["证券代码", "交易市场", "交易档位"], inplace=True, drop=True)

        # 得到当前的账户价值
        total_value = self.broker.getvalue()
        # 以往标的和本次标的重复的持仓市值
        last_ranks_values = sum(
            [
                self.broker.getvalue([i])
                for i in set(self.lastRanks) & set(self.ranks)
                if self.broker.getvalue([i])
            ]
        )
        # 总资产 - 不清仓的市值 获得剩余现金进行仓位分配
        self.p_value = (total_value * self.timing_data - last_ranks_values) / len(
            self.long_ranks
        )
        if self.p_value < 0:
            print("可分配资金小于0", self.p_value)
        # 下单
        for data in self.long_ranks:
            # 按照次日开盘价计算下单量,下单量是100的整数倍
            if len(data) < data.buflen():
                # 判断是否满足估值条件
                trade_plan = self.values_trade_plane(data)

                # 满足就买入，不满足就创建交易计划，后续执行
                if trade_plan:
                    # 必然为初次买入
                    size = int(self.p_value / 100 / data.open[1]) * 100
                    self.buy(data=data, size=size)
                else:
                    # 不满足，金字塔买入
                    trade_plan_data = self.trade_plan[
                        self.trade_plan.index.get_level_values("证券代码") == data._name
                    ]
                    assert trade_plan_data.empty, f"{data._name}已有交易计划，这是错误的！"
                    # 必然为初次买入
                    size = int(self.p_value * 0.5 / 100 / data.open[1]) * 100
                    self.buy(data=data, size=size)
                    # 创建后续买入计划
                    self.trade_plan.loc[(data._name, "XSHG", "买入点"), "买入原因"] = "金字塔"
                    self.trade_plan.loc[(data._name, "XSHG", "买入点"), "股价变动幅度"] = "-30%"
                    self.trade_plan.loc[(data._name, "XSHG", "买入点"), "买入比例"] = 0.5
                    self.trade_plan.loc[(data._name, "XSHG", "买入点"), "股价"] = data.open[
                        1
                    ] * (1 - 0.3)
                    self.trade_plan.loc[(data._name, "XSHG", "买入点"), "计划执行次数"] = int(0)
                    self.trade_plan.loc[
                        (data._name, "XSHG", "买入点"), "计划创建时间"
                    ] = data.datetime.date(0)
                    self.trade_plan.loc[(data._name, "XSHG", "买入点"), "计划可用资金"] = (
                        self.p_value * 0.5
                    )

        # 跟踪上次买入的标的
        self.lastRanks = self.ranks
        self.warehouse_date = True

        # 打印仓位信息
        self.trading_position()

    def stop(self):
        self.log_file.close()
        self.log_file_dataframe.to_excel(
            f"C:/Users/32771/Desktop/回测/风控/{self.p.strategy_name}-每日持仓.xlsx"
        )
        self.transaction_reason.to_excel(
            f"C:/Users/32771/Desktop/回测/风控/{self.p.strategy_name}-{str(self.p.values_num)}-买入成交记录.xlsx"
        )
        if self.risk_reason.empty:
            self.risk_reason = pd.DataFrame(index=[0])
        self.risk_reason.to_excel(
            f"C:/Users/32771/Desktop/回测/风控/{self.p.strategy_name}-{str(self.p.values_num)}-风控记录.xlsx"
        )

    def log(self, txt, dt=None, doprint=False):
        if self.params.printlog or doprint:
            dt = dt or self.datas[0].datetime.date(0)
            print(f"{dt.isoformat()},{txt}")

    # 记录交易执行情况（可省略，默认不输出结果）
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # 订单状态 submitted/accepted，无动作
            return

        # 订单完成
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    "买单执行,%s, %.2f, %i"
                    % (order.data._name, order.executed.price, order.executed.size)
                )
                trade_data = self.trade_plan[
                    self.trade_plan.index.get_level_values("证券代码") == order.data._name
                ]
                log_date = self.data0.datetime.date(0).strftime("%Y%m%d")
                if not trade_data.empty and trade_data.计划执行次数[0] >= 1:
                    self.transaction_reason = self.transaction_reason.append(
                        pd.DataFrame(
                            {
                                "股票代码": order.data._name,
                                "买入价格": order.executed.price,
                                "买入数量": order.executed.size,
                                "买入原因": self.trade_plan.loc[
                                    (order.data._name, "XSHG", "买入点"), "买入原因"
                                ],
                            },
                            index=[log_date],
                        )
                    )

                    # 买完应该删除计划
                    self.trade_plan = self.trade_plan.drop((order.data._name, "XSHG"))
                else:
                    self.transaction_reason = self.transaction_reason.append(
                        pd.DataFrame(
                            {
                                "股票代码": order.data._name,
                                "买入价格": order.executed.price,
                                "买入数量": order.executed.size,
                                "买入原因": "股票池入选",
                            },
                            index=[log_date],
                        )
                    )

            elif order.issell():
                self.log(
                    "卖单执行, %s, %.2f, %i"
                    % (order.data._name, order.executed.price, order.executed.size)
                )
        else:
            self.log(
                "订单作废 %s, %s, isbuy=%i, size %i, open price %.2f"
                % (
                    order.data._name,
                    order.getstatusname(),
                    order.isbuy(),
                    order.created.size,
                    order.data.open[0],
                )
            )

    # 记录交易收益情况（可省略，默认不输出结果）
    # def notify_trade(self, trade):
    #     if trade.isclosed:
    #         print('毛收益 %0.2f, 扣佣后收益 % 0.2f, 佣金 %.2f, 市值 %.2f, 现金 %.2f' %
    #               (trade.pnl, trade.pnlcomm, trade.commission, self.broker.getvalue(), self.broker.getcash()))


def test_start():
    def apply_code(codes):
        return codes[:6]

    import time
    import os
    from datetime import date

    def file_name(file_dir):
        for root, dirs, files in os.walk(file_dir):
            return root, dirs, files

    root_dir, sub_dirs, files = file_name("C:/Users/32771/Desktop/回测/风控/图片")
    relocation_date = "0501"
    for ye in ["20100101"]:
        if files != []:
            # if ye in [i.split('-')[0] for i in files]:
            if ye in files[0][:8]:
                continue
        strategy_name = f"{ye}"
        start = f"{ye}"
        end = "20210616"
        res_df = []
        trade_data = StockTradeDay.query_data(
            start_timestamp=pd.to_datetime(start) - np.timedelta64(1, "Y"),
            end_timestamp=pd.to_datetime(end),
        ).timestamp
        trade_data = list(set(str(i.year) for i in trade_data))
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # code_dict = {}
        # buy_signal = pd.read_csv("C:/Users/32771/Documents/dev data/buy_signal.csv")
        # buy_signal['timestamp'] = pd.to_datetime(buy_signal['timestamp'])
        # buy_signal = buy_signal[buy_signal['timestamp'] >= pd.to_datetime(start) - np.timedelta64(1, "Y")]
        # buy_signal['股票代码'] = buy_signal['股票代码'].apply(lambda x:str(x).zfill(6))
        #
        buy_signal = pd.DataFrame()
        no_symbol = []
        for sheet_name in trade_data:
            data = pd.read_excel(
                "C:/Users/32771/Documents/dev data/0510/副本副本2005至今_ROE变异值2.xlsx",
                sheet_name=sheet_name,
            )
            data = data[~data.股票代码.isin(no_symbol)].iloc[:10]
            data["year"] = sheet_name
            buy_signal = buy_signal.append(data[["year", "股票代码", "股票名称", "股票池"]])

        buy_signal = buy_signal.applymap(lambda x: apply_code(str(x)))
        buy_signal["timestamp"] = buy_signal.year.apply(
            lambda x: StockTradeDay.query_data(
                start_timestamp=str(x) + relocation_date, limit=1
            ).timestamp.values[0]
            if str(x) + relocation_date
            <= date.fromtimestamp(time.time()).strftime("%Y%m%d")
            else np.NaN
        )

        buy_signal["股票代码"] = buy_signal.股票代码.replace("000022", "001872")
        buy_signal["股票名称"] = buy_signal.股票名称.replace("深赤湾A", "招商港口")
        buy_signal = buy_signal.query("timestamp <= @end")
        code_dict = {}
        for stock_data in buy_signal[["股票代码", "timestamp", "股票名称", "股票池"]].to_dict(
            "records"
        ):
            stock_n = stock_data["股票代码"] + "_" + stock_data["股票名称"]
            if stock_n not in code_dict:
                code_dict.update({stock_n: stock_data["timestamp"]})
            else:
                if code_dict[stock_n] >= stock_data["timestamp"]:
                    code_dict.update({stock_n: stock_data["timestamp"]})
        data_block = BlockStock.query_data(
            filters=[
                BlockStock.stock_code.in_(list(set(buy_signal["股票代码"].tolist()))),
                BlockStock.block_type == "swl1",
            ]
        )[["stock_code", "name"]]
        buy_signal = pd.merge(
            data_block, buy_signal, left_on=["stock_code"], right_on=["股票代码"]
        )
        buy_signal["CODES"] = (
            buy_signal["股票代码"] + "_" + buy_signal["股票名称"] + "_" + buy_signal["name"]
        )
        buy_signal.set_index(["timestamp"], drop=False, inplace=True)
        # 风控数据
        risk_start = pd.to_datetime(start) - np.timedelta64(1, "Y")
        risk_end = pd.to_datetime(end)
        risk_stock_list = list(set(buy_signal["股票代码"].tolist()))
        HolderTradingData = HolderTrading.query_data(
            filters=[HolderTrading.holder_direction == "减持"],
            codes=risk_stock_list,
            start_timestamp=risk_start,
            end_timestamp=risk_end,
            provider="emquantapi",
        )

        # 收入中应收账款占比不能超过50
        # 质押比例
        EquityPledgeData = EquityPledge.query_data(
            filters=[
                EquityPledge.end_date >= risk_start,
            ],
            columns=["pledge_total_ratio", "start_date", "end_date", "code"],
            codes=risk_stock_list,
        )

        # 应收账款
        BalanceSheetData = BalanceSheet.query_data(
            filters=[BalanceSheet.report_period == "year"],
            columns=["accounts_receivable", "report_date", "code"],
            codes=risk_stock_list,
            end_timestamp=risk_end,
        )
        # 营业收入
        IncomeStatementData = IncomeStatement.query_data(
            filters=[IncomeStatement.report_period == "year"],
            columns=["operating_income", "report_date", "code"],
            codes=risk_stock_list,
            end_timestamp=risk_end,
        )

        code_dict.update({"000001_上证指数_指数": pd.to_datetime(start)})
        # for vlnum in [100, 80, 50, 60, 40, 20]:
        for vlnum in [50]:
            st1 = time.time()
            cerebro = bt.Cerebro(tradehistory=True)
            for name in sorted(code_dict):
                # if i >= 10:
                #     break
                # data = save_csv(name,start,end)

                data = pd.read_csv(
                    f"C:/Users/32771/Documents/dev data/0617/datachoices/合并/{name[:6]}.csv"
                )

                data["date"] = pd.to_datetime(data["date"])
                data.set_index(["date"], drop=True, inplace=True)
                data = data[
                    [
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "pe_ttm_分位",
                        "peg",
                        "pcf_分位",
                        "ps_ttm_分位",
                        "pb_分位",
                        "openinterest",
                    ]
                ].query("date >= @start")
                if name != "000001_上证指数_指数":
                    name = buy_signal[buy_signal.股票代码 == name[:6]].CODES.iloc[0]

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
                    todate=pd.to_datetime(data.index[-1].strftime("%Y%m%d")),  # 结束日
                )
                print(name)
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
            cerebro.addanalyzer(Trade_list, _name="trade_list")
            cerebro.broker.set_checksubmit(False)

            # 添加策略
            cerebro.addstrategy(
                MyStrategy,
                strategy_name=strategy_name,
                values_num=vlnum,
                buy_signal=buy_signal,
                HolderTradingData=HolderTradingData,
                EquityPledgeData=EquityPledgeData,  # 质押比例数据
                BalanceSheetData=BalanceSheetData,  #
                IncomeStatementData=IncomeStatementData,  # 应收账款数据
                printlog=True,
            )
            df00, df0, df1, df2, df3, df4, trade_list_df = bt.out_result(cerebro)
            # cerebro.run()
            # b=Bokeh(style='bar',tabs='multi',sched=Tradimo()) #传统白底，多页
            # cerebro.plot(b)
            df0[f"{vlnum}%分位低估"] = df0[f"total_value"]
            df0[[f"{vlnum}%分位低估"]].to_excel(
                f"C:/Users/32771/Desktop/回测/风控/{strategy_name}-{vlnum}-估值表现.xlsx"
            )
            res_df.append(df0[[f"{vlnum}%分位低估"]])
            trade_list_df.rename(columns=trade_col, inplace=True)
            trade_list_df.to_excel(
                f"C:/Users/32771/Desktop/回测/风控/{strategy_name}-{vlnum}-交易详情.xlsx"
            )
            df1.to_excel(
                f"C:/Users/32771/Desktop/回测/风控/{strategy_name}-{vlnum}-绩效指标.xlsx"
            )
            # 绩效指标
            # df00.绩效指标.replace({
            #     'calmar_ratio':'卡玛比率',
            #     'average_drawdown_len':'平均回撤长度',
            #     'average_drawdown_rate':'平均回撤比率',
            #     'average_drawdown_money':'平均回撤资金',
            #     'max_drawdown_len':'最大回撤长度',
            #     'max_drawdown_rate':'最大回撤比率',
            #     'average_rate':'平均收益率',
            #     'stddev_rate':'标准差',
            #     'positive_year':'获利年份',
            #     'negative_year':'亏损年份',
            #     'nochange_year':'无变化年份',
            #     'best_year':'最大年收益率',
            #     'worst_year':'最差年收益率',
            #     'sqn_ratio':'系统质量指数',
            #     'vwr_ratio':'变异加权回报',
            #     'sharpe_info':'夏普比率',
            # })
            df00.to_excel(
                f"C:/Users/32771/Desktop/回测/风控/{strategy_name}-{vlnum}-GrossLeverage.xlsx"
            )
            end1 = time.time()
            print(vlnum, " ", (end1 - st1) / 60, "分钟")
            del df0, vlnum


def save_csv(name, start, end):
    # data = pd.read_csv(f"C:/Users/32771/Documents/dev data/0604/行情/{name[:6]}.csv")
    if name[:6] == "000001":
        data = get_index_data(start, end)
    else:
        data = get_data_back(name[:6], start, end)
    data["date"] = data.index
    data["date"] = pd.to_datetime(data["date"])
    data.reset_index(drop=True, inplace=True)
    # 000022 - 估值, 没数据
    if name[:6] != "000001":
        try:
            data_values = pd.read_csv(
                f"C:/Users/32771/Documents/dev data/0617/datachoices/行业分类/"
                f"{name[:6]}.csv"
            )
            data_values["date"] = pd.to_datetime(data_values["date"])
            data_values["peg"] = data_values["PEG"]
        except:
            print(f"{name[:6]}-估值,没数据")
            return
        data_values["date"] = pd.to_datetime(data_values["date"])
        name = name + "_" + data_values["行业分类"].iloc[0]
    else:
        data_values = pd.DataFrame(
            columns=["pe_ttm_分位", "peg", "pcf_分位", "pb_分位", "ps_ttm_分位", "行业分类"]
        )
        data_values["date"] = data["date"]

    data = pd.merge(data_values, data, on=["date"])
    data = data[
        [
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "pe_ttm_分位",
            "peg",
            "pcf_分位",
            "ps_ttm_分位",
            "pb_分位",
            "openinterest",
        ]
    ]
    data.to_csv(f"C:/Users/32771/Documents/dev data/0617/datachoices/合并/{name[:6]}.csv")
    data.set_index(["date"], drop=True, inplace=True)
    print(name)
    return data


test_start()

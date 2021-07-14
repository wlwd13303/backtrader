import pandas as pd
import numpy as np
import pymysql
import talib as ta
from datetime import datetime, timedelta
# zvt

zvthost='123.103.74.231'
zvtuser = 'zvtreader'
zvtpasswd = 'zvtreader321'
zvtdb = 'zvt'
zvtport = 3306

def timing_robot(symbol: str, start: int, end: int) -> pd.DataFrame:
    """ 入口函数
    :param symbol: str,指数代码
    :param start:int,开始日期
    :param end:int,结束日期
    :return:
    """
    assert isinstance(start, int) and isinstance(end, int), 'start,end,symbo入参必须为int类型'
    assert isinstance(symbol, str), 'symbo入参必须为str类型'
    start, end = str(start), str(end)

    start1 = (datetime.strptime(start, '%Y%m%d') - timedelta(days=365)).strftime('%Y%m%d')
    ddf = read_data(symbol, start1, end)
    ddf = pre(ddf)
    result_data = period_position(ddf)
    date_con = result_data['tdate'].map(lambda x: x >= pd.to_datetime(start) and x <= pd.to_datetime(end))
    result_data = result_data[date_con]
    result_data = result_data[
        ['tdate', 'tclose', 'market_trend_shape', 'position_rate_advice', 'position_rate', 'operation_advice']]
    return result_data


def read_data(symbol, start, end):
    """
    读取指数行情数据
    """
    ddf = get_daily_index([symbol], str(start), str(end))
    ddf = ddf.sort_values(by='TDATE')
    return ddf

def get_daily_index(symbol_list: list, start_val: str, end_val: str):
    """
    获取指数前复权日行情
    """
    conn = pymysql.connect(host=zvthost, user=zvtuser, passwd=zvtpasswd, db=zvtdb,
                           port=zvtport)

    sql1 = 'select * from index_1d_kdata ' \
           'where  index_1d_kdata.timestamp<= %s ' \
           'and index_1d_kdata.timestamp >= %s and ' \
           'code in ({})'.format(",".join(["'%s'" % item for item in symbol_list]))

    daily_stock = pd.read_sql(sql1, conn, params=(str(end_val), str(start_val)))
    conn.close()
    daily_stock.rename(columns={
        "code": "SYMBOL",
        "timestamp": "TDATE",
        "close": "TCLOSE",
        "low": "LOW",
        "high": "HIGH",
        "open": "TOPEN",
        "volume": "VOTURNOVER",  # 成交量
        "turnover": "VATURNOVER",  # 成交额
    }, inplace=True)
    daily_stock["EXCHANGE"] = daily_stock.entity_id.str.split("_", expand=True)[1]
    daily_stock["EXCHANGE"].replace({"sz": "CNSESZ", "sh": "CNSESH"}, inplace=True)
    daily_stock["LCLOSE"] = daily_stock.sort_values("TDATE").groupby("entity_id").TCLOSE.shift(1)
    return daily_stock


def pre(ddf):
    """
    计算通道和相关指标
    """
    ema_list1 = [5, 10, 20, 60]
    short = 20
    limit = 0.005

    for num in ema_list1:
        ddf['EMA_' + str(num)] = ta.EMA(np.array(ddf['TCLOSE']), timeperiod=num)  # 计算不同周期的EMA均线

    ddf["avg_ema"] = np.mean(np.array(ddf[['EMA_5','EMA_10','EMA_20','EMA_60']]), axis=1)  # 对均线求均值
    ddf['middle'] = ddf['avg_ema'].rolling(short).mean()
    ddf['up_limit'] = ddf['middle'] * (1 + limit)  # 通道上轨
    ddf['down_limit'] = ddf['middle'] * (1 - limit)  # 通道下轨

    ddf = calc_index(ddf)
    ddf = ddf.dropna()
    ddf = ddf.reset_index(drop=True)
    return ddf


def calc_index(data):
    """
    计算MACD、WR和RSI指标
    """
    data['DIF'], data['DEA'], data['MACD'] = ta.MACD(np.array(data['TCLOSE']))  # 计算MACD
    data['hhv_macd'] = data['MACD'].rolling(5).max()

    data['hhv'] = data['HIGH'].rolling(6).max()  # 计算WR指标
    data['llv'] = data['LOW'].rolling(6).min()
    data['wr'] = 100 * (data['hhv'] - data['TCLOSE']) / (data['hhv'] - data['llv'])

    data['rsi'] = ta.RSI(np.array(data['TCLOSE']), timeperiod=6)  ##计算RSI
    return data


def period_position(ddf):
    """
    计算仓位
    """
    close_list = np.array(ddf['TCLOSE'])
    high_list = np.array(ddf['HIGH'])
    low_list = np.array(ddf['LOW'])
    up_limit_list = np.array(ddf['up_limit'])
    down_limit_list = np.array(ddf['down_limit'])
    wr_list = np.array(ddf['wr'])
    rsi_list = np.array(ddf['rsi'])
    dif_list = np.array(ddf['DIF'])
    dea_list = np.array(ddf['DEA'])
    hhv_macd_list = np.array(ddf['hhv_macd'])
    macd_list = np.array(ddf['MACD'])

    # 初始化信号
    sig = 0
    signal = 2
    position = 0
    position_advice = '0-20%'
    trend = '下降'
    operation_advice = 1

    # 存储中间信号
    down_wr_down_list = []
    down_rsi_up_list = []
    down_wr_rsi_buy_list = []

    up_wr_down_list = []
    up_rsi_up_list = []
    up_wr_rsi_buy_list = []

    dif_down_list = []  # DIF向下拐头
    dea_cross_dif_list = []  # DIF死叉DEA
    dif_cross_dea_list = []  # DIF金叉DEA
    dif_dea_cross_list = []  # DIF死叉DEA或DIF金叉DEA

    # 存储结果
    position_list = []  # 仓位
    position_advice_list = []  # 仓位建议
    trend_list = []  # 市场趋势形态
    operation_advice_list = []  # 操作建议的唯一标识符

    for i in range(len(ddf)):
        up_wr_down_list.append(value_cross(70, wr_list, i) and turn_down(wr_list, i))  # 向下拐头+下穿固定值[上通道]
        up_rsi_up_list.append(cross_value(rsi_list, 50, i) and turn_up(rsi_list, i))  # 向上拐头+上穿固定值[上通道]
        up_wr_rsi_buy_list.append(up_wr_down_list[i] and up_rsi_up_list[i])  # [上通道]

        down_wr_down_list.append(value_cross(60, wr_list, i) and turn_down(wr_list, i))  # 向下拐头+下穿固定值[下通道]
        down_rsi_up_list.append(cross_value(rsi_list, 35, i) and turn_up(rsi_list, i))  # 向上拐头+上穿固定值[下通道]
        down_wr_rsi_buy_list.append(down_wr_down_list[i] and down_rsi_up_list[i])

        dif_down_list.append(turn_down(dif_list, i))
        dif_cross_dea_list.append(cross(dif_list, dea_list, i))
        dea_cross_dif_list.append(cross(dea_list, dif_list, i))
        dif_dea_cross_list.append(dif_cross_dea_list[i] or dea_cross_dif_list[i])

        cur_interval_deviate_up = interval_deviate_up(macd_list, dif_dea_cross_list, high_list, dif_list, i)  # 顶背离
        cur_cross_top = cross_top(dea_cross_dif_list, dif_cross_dea_list, close_list, dif_list, high_list, i)  # 交叉顶背离
        cur_macd_pillar_top = macd_pillar_top(dea_cross_dif_list, dif_cross_dea_list, dea_list, close_list,
                                              hhv_macd_list, low_list, i)  # 柱顶背离
        cur_apart_interval_deviate_up = apart_interval_deviate_up(macd_list, dif_dea_cross_list, high_list, dif_list,
                                                                  i)  # 隔峰顶背离
        if signal == 2 and cross(close_list, up_limit_list, i):  # 收盘价上穿上轨
            sig = 1
            signal = 1
            position = 1
            trend = '上升'
            position_advice = '70-100%'
            operation_advice = 8
        elif signal == 1 and cross(down_limit_list, close_list, i):  # 收盘价下穿下轨
            sig = 2
            signal = 2
            position = 0
            trend = '下降'
            position_advice = '0-20%'
            operation_advice = 1

        if signal == 1 and up_wr_rsi_buy_list[i]:  # 在上轨道上，wr_rsi双指标发出买入信号
            sig = 3;
            buy_low = low_list[i]
            position = 1;
            trend = '上升'
            position_advice = '70-100%'
            operation_advice = 8

        elif signal == 1 and dif_list[i] > 0 and cur_interval_deviate_up[0] and dif_down_list[i]:  # 顶背离
            sig = 6
            sell_high = cur_interval_deviate_up[1]
            position = 0
            trend = '下降'
            position_advice = '0-20%'
            operation_advice = 1

        elif signal == 1 and dif_list[i] > 0 and cur_apart_interval_deviate_up[0] and dif_down_list[i]:  # 隔峰顶背离
            sig = 6
            sell_high = cur_apart_interval_deviate_up[1]
            position = 0
            trend = '下降'
            position_advice = '0-20%'
            operation_advice = 1

        elif signal == 1 and dif_list[i] > 0 and cur_macd_pillar_top[0]:  # 柱顶背离
            sell_high = cur_macd_pillar_top[1]
            sig = 6
            position = 0
            trend = '下降'
            position_advice = '0-20%'
            operation_advice = 1

        elif signal == 1 and dif_list[i] > 0 and dif_list[i] > 0 and cur_cross_top[0]:  # 交叉背离
            sell_high = cur_cross_top[1]
            sig = 6
            position = 0
            trend = '下降'
            position_advice = '0-20%'
            operation_advice = 1

        elif signal == 1 and sig == 6 and close_list[i] > sell_high:  # 背离判断有误
            sig = 7
            position = 1
            trend = '上升'
            position_advice = '70-100%'
            operation_advice = 8

        elif signal == 2 and down_wr_rsi_buy_list[i]:  # 在下轨道上，wr_rsi双指标发出买入信号
            sig = 8
            buy_low = low_list[i]
            position = 0.5
            trend = '反弹'
            position_advice = '30-50%'
            operation_advice = 3

        elif signal == 2 and sig == 8 and close_list[i] < buy_low:  # wr_rsi双指标买入判断有误
            sig = 9
            position = 0
            trend = '下降'
            position_advice = '0-20%'
            operation_advice = 1

        position_list.append(position)
        position_advice_list.append(position_advice)
        trend_list.append(trend)
        operation_advice_list.append(operation_advice)
    result_data = pd.DataFrame(
        {'tdate': ddf['TDATE'], 'market_trend_shape': trend_list, 'operation_advice': operation_advice_list,
         'position_rate_advice': position_advice_list, 'position_rate': position_list,
         'tclose': ddf['TCLOSE']})

    return result_data


def cross(dif, dea, i):
    """
    金叉
    """
    cond = False
    if i > 0 and dif[i - 1] <= dea[i - 1] and dif[i] >= dea[i]:
        cond = True
    return cond


def cross_value(data1, num, i):
    """
    上穿固定值
    """
    cond = False
    if i > 0 and data1[i] > num and data1[i - 1] < num:
        cond = True
    return cond


def value_cross(num, data1, i):
    """
    下穿固定值
    """
    cond = False
    if i > 0 and data1[i] < num and data1[i - 1] > num:
        cond = True
    return cond


def turn_down(data, i):
    """
    向下拐头
    """
    cond = False
    if i > 1 and data[i] < data[i - 1] and data[i - 1] > data[i - 2]:
        cond = True
    return cond


def turn_up(data, i):
    """
    向上拐头
    """
    cond = False
    if i > 1 and data[i - 2] > data[i - 1] and data[i - 1] < data[i]:
        cond = True
    return cond


def interval_deviate_up(macd_list, k_d_cross_list, high_list, k_list, m):
    """
    不隔峰顶背离：未交叉时判断，high_list创新高，k_list没有创新高
    """
    cond = False
    period_max = 10 ** 8
    try:
        k_cross_d_list1 = k_d_cross_list[m::-1]
        k_first = k_cross_d_list1.index(True)
        k_second = k_cross_d_list1[k_first + 1:].index(True)
        k_three = k_cross_d_list1[k_first + k_second + 2:].index(True)
    except ValueError:
        pass
    else:
        k_first = m - k_first
        k_second = k_first - k_second - 1
        k_three = k_second - k_three - 1
        if macd_list[m] > 0 and (max(high_list[:m + 1][k_first:m + 1]) > max(high_list[:m + 1][k_three:k_second])) and (
                max(k_list[:m + 1][k_first:m + 1]) < max(k_list[:m + 1][k_three:k_second])):
            cond = True
            period_max = max(high_list[:m + 1][k_first:m + 1])
    return [cond, period_max]


def apart_interval_deviate_up(macd_list, k_d_cross_list, high_list, k_list, m):
    """
    定义隔峰顶背离：未交叉时判断，high_list创新高，k_list没有创新高
    """
    cond = False
    period_max = 10 ** 8
    try:
        k_cross_d_list1 = k_d_cross_list[m::-1]
        k_first = k_cross_d_list1.index(True)
        k_second = k_cross_d_list1[k_first + 1:].index(True)
        k_three = k_cross_d_list1[k_first + k_second + 2:].index(True)
        k_four = k_cross_d_list1[k_first + k_second + k_three + 3:].index(True)
        k_five = k_cross_d_list1[k_first + k_second + k_three + k_four + 4:].index(True)
    except ValueError:
        pass
    else:
        k_first = m - k_first
        k_second = k_first - k_second - 1
        k_three = k_second - k_three - 1
        k_four = k_three - k_four - 1
        k_five = k_four - k_five - 1
        if macd_list[m] > 0 and (max(high_list[:m + 1][k_first:m + 1]) > max(high_list[:m + 1][k_five:k_four])) and (
                max(k_list[:m + 1][k_first:m + 1]) < max(k_list[:m + 1][k_five:k_four])):  # 股票在区间内的最低价的最低价创新低
            cond = True
            period_max = max(high_list[:m + 1][k_first:m + 1])
    return [cond, period_max]


def macd_pillar_top(k_cross_d_list, d_cross_k_list, dea_list, close_list, hhv_macd_list, high_list, m):
    """
    定义柱顶背离
    """
    cond = False
    period_max = 10 ** 8
    try:
        k_cross_d_list1 = k_cross_d_list[m::-1]
        d_cross_k_list1 = d_cross_k_list[m::-1]
        k_first = k_cross_d_list1.index(True)
        k_second = k_cross_d_list1[k_first + 1:].index(True)
        d_first = d_cross_k_list1.index(True)
    except ValueError:
        pass
    else:
        k_first = m - k_first
        k_second = k_first - k_second - 1
        d_first = m - d_first
        if k_cross_d_list[m] and dea_list[m] > 0 and close_list[k_second] > close_list[k_first] and hhv_macd_list[
            k_second] < hhv_macd_list[k_first]:
            cond = True
            period_max = max(high_list[d_first:k_first])
    return [cond, period_max]


def cross_top(k_cross_d_list, d_cross_k_list, close_list, hhv_macd_list, high_list, m):
    """
    定义交叉顶背离：交叉时判断，close_list创新高，hhv_macd_list没有创新高
    """
    cond = False
    period_max = 10 ** 8
    try:
        k_cross_d_list1 = k_cross_d_list[m::-1]
        d_cross_k_list1 = d_cross_k_list[m::-1]
        k_first = k_cross_d_list1.index(True)
        k_second = k_cross_d_list1[k_first + 1:].index(True)
        d_first = d_cross_k_list1.index(True)
    except ValueError:
        pass
    else:
        k_first = m - k_first
        k_second = k_first - k_second - 1
        d_first = m - d_first
        if k_cross_d_list[m] and close_list[k_second] < close_list[k_first] and hhv_macd_list[k_second] > hhv_macd_list[
            k_first]:
            cond = True
            period_max = max(high_list[d_first:k_first])
    return [cond, period_max]

if __name__ == '__main__':
    load_config('/home/wanglei/conf')
    start = 20210420
    end = 20210420
    symbol = '000300'
    df = timing_robot(symbol, start, end)
    import stralib

import datetime

import requests
import json
import pandas as pd


def values_trade_plane(data, values_num):
    """
    根据股票所处行业，判断其估值是否满足买点
    Args:
        data:

    Returns:

    """
    trade_plan = False
    if val_df['CODES'].values[0].split('_')[-1] == '电子I' \
            and (data.peg.values[0] < 1 or data.ps_ttm_分位.values[0] < values_num or data.pb_分位.values[0] < values_num):
        trade_plan = True
    elif val_df['CODES'].values[0].split('_')[-1] in ['家用电器I', '电气设备I'] \
            and (data.peg.values[0] < 1 or data.pcf_分位.values[0] < values_num or data.ps_ttm_分位.values[0] < values_num):
        trade_plan = True
    elif val_df['CODES'].values[0].split('_')[-1] == '食品饮料I' \
            and (data.peg.values[0] < 1 or data.pb_分位.values[0] < values_num):
        trade_plan = True
    elif val_df['CODES'].values[0].split('_')[-1] == '医药生物I' \
            and (data.peg.values[0] < 1 or data.pb_分位.values[0] < values_num):
        trade_plan = True
    elif val_df['CODES'].values[0].split('_')[-1] == '交通运输I' \
            and (data.pcf_分位.values[0] < values_num or data.pb_分位.values[0] < values_num):
        trade_plan = True
    elif val_df['CODES'].values[0].split('_')[-1] == '化工I' \
            and (data.peg.values[0] < 1 or data.pcf_分位.values[0] < values_num or data.ps_ttm_分位.values[0] < values_num):
        trade_plan = True
    elif val_df['CODES'].values[0].split('_')[-1] == '房地产I' \
            and (data.peg.values[0] < 1 or data.pcf_分位.values[0] < values_num or data.ps_ttm_分位.values[0] < values_num):
        trade_plan = True
    # 信息服务
    elif val_df['CODES'].values[0].split('_')[-1] == '计算机I' \
            and (
            data.pb_分位.values[0] < values_num or data.pe_ttm_分位.values[0] < values_num or data.ps_ttm_分位.values[
        0] < values_num):
        trade_plan = True
    elif val_df['CODES'].values[0].split('_')[-1] == '轻工制造I' \
            and (data.peg.values[0] < 1 or data.ps_ttm_分位.values[0] < values_num):
        trade_plan = True
    elif val_df['CODES'].values[0].split('_')[-1] == '采掘I' \
            and (data.pe_ttm_分位.values[0] < values_num or data.pcf_分位.values[0] < values_num):
        trade_plan = True
    return trade_plan


beehive3_api = "https://api.jinniuai.com"
beehive3_user = {
    "user": {
        "mobile": "18733318168",
        "api_key": "beehive3 master",
        "password": "JinChongZi321"
    }
}

response = requests.post(f"{beehive3_api}/api/auth/login", json=beehive3_user)
logined_root_user = response.json()
token = logined_root_user["token"]

s = requests.Session()
s.headers.update({"Authorization": f"Token {token}"})

# 获取组合列表
response = s.get(f"{beehive3_api}/api/portfolio")
portfolio_id_list = [rj["_id"] for rj in response.json()["data"] if rj["username"] == beehive3_user["user"]["mobile"]]

portfolio_id = portfolio_id_list[0]
# 获取组合基本信息
response = s.get(f"{beehive3_api}/api/portfolio/basic_run_data/{portfolio_id}")
data = response.json()
# 获取组合资产
account_asset_response = s.get(f"{beehive3_api}/api/portfolio/account_asset/{portfolio_id}")
account_asset_response.json()
# 获取组合持仓
account_stock_position_response = s.get(f"{beehive3_api}/api/portfolio/position/{portfolio_id}")
account_position = account_stock_position_response.json()
# 现有持仓
stock_df = pd.DataFrame()
for sotck_pos in account_position['industry_info']:
    for i in sotck_pos['stocks']:
        stock_df = stock_df.append(pd.DataFrame(i, index=[0]))

import pandas as pd
import numpy as np
import os

from EmQuantAPI import *

# 登录Choice接口
loginResult = c.start("ForceLogin=1")

from zvt.domain import *
import time
from datetime import date
import pandas as pd
import numpy as np


# 13501148196


def apply_code(codes):
    return codes[:6]


back_date = ['20050101']
for ye in back_date:
    strategy_name = f'{ye}_被选次数2'
    start = f'{ye}'
    end = '20210701'
    trade_data = StockTradeDay.query_data(start_timestamp=pd.to_datetime(start) - np.timedelta64(1, 'Y'),
                                          end_timestamp=pd.to_datetime(end)).timestamp
    trade_data = list(set(str(i.year) for i in trade_data))

    # 设置调仓日
    relocation_date = '0501'
    buy_signal = pd.DataFrame()
    no_symbol = []
    for sheet_name in trade_data:
        data = pd.read_excel("C:/Users/32771/Documents/dev data/0510/副本副本2005至今_ROE变异值2.xlsx", sheet_name=sheet_name)
        data = data[~data.股票代码.isin(no_symbol)].iloc[:10]
        data['year'] = sheet_name
        buy_signal = buy_signal.append(data[['year', '股票代码', '股票名称', '股票池']])

    buy_signal = buy_signal.applymap(lambda x: apply_code(str(x)))
    buy_signal['timestamp'] = buy_signal.year.apply(
        lambda x: StockTradeDay.query_data(start_timestamp=str(x) + relocation_date, limit=1).timestamp.values[
            0] if str(x) + relocation_date <= date.fromtimestamp(time.time()).strftime("%Y%m%d") else np.NaN)
    buy_signal = buy_signal.query("timestamp <= @end")
    code_dict = {}
    for stock_data in buy_signal[['股票代码', 'timestamp', '股票名称', '股票池']].to_dict('records'):
        stock_n = stock_data['股票代码'] + '_' + stock_data['股票名称']
        if stock_n not in code_dict:
            code_dict.update({stock_n: stock_data['timestamp']})
        else:
            if code_dict[stock_n] >= stock_data['timestamp']:
                code_dict.update({stock_n: stock_data['timestamp']})
buy_signal['CODES'] = buy_signal['股票代码'] + '_' + buy_signal['股票名称']
list(set(buy_signal['股票代码'].tolist()))
data = BlockStock.query_data(
    filters=[BlockStock.stock_code.in_(list(set(buy_signal['股票代码'].tolist()))), BlockStock.block_type == 'swl1'])
start = '20050101'
end = '20210701'

for index, df in data[['stock_code', 'name', 'stock_id']].iterrows():
    em_id = str(df.stock_id).split('_')
    exchange = 'SH' if 'sh' in em_id[1] else 'SZ'
    em_code = em_id[-1] + '.' + exchange
    print(em_code)

    old_data = pd.read_csv(f"C:/Users/32771/Desktop/回测/实盘/估值/{df.stock_code}.csv")
    del old_data['Unnamed: 0']
    old_data['DATES'] = pd.to_datetime(old_data['DATES'])
    old_end = old_data['DATES'].max().strftime("%Y%m%d")
    if old_end == end:
        print(em_code, "em_code")

    data_peg = c.csd(em_code, "PCF,PETTM,PSTTM,PB,PEG", old_end, end,
                     "Type=1,period=1,adjustflag=1,curtype=1,order=1,market=CNSESH,ispandas=1")
    data_peg.reset_index(drop=False, inplace=True)
    data_peg['DATES'] = pd.to_datetime(data_peg.DATES)
    data_peg['行业分类'] = df['name']
    old_data = old_data.append(data_peg)
    old_data.sort_values("DATES", inplace=True)
    old_data.reset_index(drop=True, inplace=True)
    old_data.to_csv(f"C:/Users/32771/Desktop/回测/实盘/估值/{df.stock_code}.csv")


def file_name(file_dir):
    """
    获取入参路径中的所有文件名
    Args:
        file_dir:

    Returns:

    """
    for root, dirs, files in os.walk(file_dir):
        return root, dirs, files


def quantile_values(df, df_all, values_name):
    """
    计算入参的百分位，滚动5年
    Args:
        df:
        df_all:
        values_name:

    Returns:

    """
    if pd.to_datetime(df.date) <= pd.to_datetime('20100101'):
        return np.NaN
    df_all = df_all.loc[df.date - np.timedelta64(5, 'Y'):df.date]
    rank_data = df_all[values_name].rank() + 1
    try:
        return (rank_data / rank_data.max()).iloc[-1] * 100
    except:
        print('错误')


for index, df in data[['stock_code', 'name', 'stock_id']].iterrows():
    em_id = str(df.stock_id).split('_')
    exchange = 'SH' if 'sh' in em_id[1] else 'SZ'
    em_code = em_id[-1] + '.' + exchange
    print(em_code)
    data_atom = pd.read_csv(f'C:/Users/32771/Desktop/回测/实盘/估值/{em_id[-1]}.csv')
    del data_atom['Unnamed: 0']
    data_atom['DATES'] = pd.to_datetime(data_atom['DATES'])
    data_atom['date'] = data_atom['DATES']
    data_atom.set_index('date', drop=False, inplace=True)
    data_atom.dropna(subset=['PCF'], inplace=True)
    data_atom['pcf_分位'] = data_atom.apply(lambda x: quantile_values(x, data_atom, 'PCF'), axis=1)
    data_atom['pe_ttm_分位'] = data_atom.apply(lambda x: quantile_values(x, data_atom, 'PETTM'), axis=1)
    data_atom['ps_ttm_分位'] = data_atom.apply(lambda x: quantile_values(x, data_atom, 'PSTTM'), axis=1)
    data_atom['pb_分位'] = data_atom.apply(lambda x: quantile_values(x, data_atom, 'PB'), axis=1)
    data_atom['peg_分位'] = data_atom.apply(lambda x: quantile_values(x, data_atom, 'PEG'), axis=1)
    data_atom.to_csv(f'C:/Users/32771/Desktop/回测/实盘/估值分位/{em_id[-1]}.csv')

buy_signaldata = buy_signal[buy_signal.year == buy_signal.year.max()]
data = BlockStock.query_data(
    filters=[BlockStock.stock_code.in_(list(set(buy_signal['股票代码'].tolist()))), BlockStock.block_type == 'swl1'])
data["股票代码"] = data['stock_code']
data["行业分类"] = data['name']

data = data[["行业分类", "股票代码"]]

buy_signaldata = pd.merge(data, buy_signaldata, on=['股票代码'])
buy_signaldata['CODES'] = buy_signaldata.apply(lambda x: x.CODES + '_' + x.行业分类, axis=1)
end = pd.to_datetime('20210624')
total_capital = account_asset_response.json()['total_capital']
p_values = total_capital * 0.95 / buy_signaldata.shape[0]

print(response.text)
"""
600563 金字塔下单
002415 金字塔下单
600519 金字塔下单
603288 金字塔下单
600436 金字塔下单
600763 金字塔下单
"""
# 开始创建交易计划
for index, df in pd.DataFrame(buy_signaldata).iterrows():
    val_df = pd.read_csv(f'C:/Users/32771/Desktop/回测/实盘/估值分位/{df.股票代码}.csv')
    val_df['date'] = pd.to_datetime(val_df['date'])
    val_df['peg'] = val_df['PEG']
    val_df = val_df.query("date == @end")
    val_df['CODES'] = df['CODES']
    if val_df.CODES.values[0][:6] == '300482':
        continue
    va_plane = values_trade_plane(val_df, 50)
    if va_plane:
        exchangedata = '1'
        data = {"symbol": val_df.CODES.values[0][:6], "exchange": exchangedata}
        response = s.get(f"{beehive3_api}/api/stock/five_real_time_price",
                         headers={"Authorization": " ".join(["Token", logined_root_user["token"]])}, params=data)
        if response.status_code != 200:
            exchangedata = '0'
            data = {"symbol": val_df.CODES.values[0][:6], "exchange": exchangedata}
            response = s.get(f"{beehive3_api}/api/stock/five_real_time_price",
                             headers={"Authorization": " ".join(["Token", logined_root_user["token"]])}, params=data)
            five_real_time_pricedata = response.json()
        else:
            response = s.get(f"{beehive3_api}/api/stock/five_real_time_price",
                             headers={"Authorization": " ".join(["Token", logined_root_user["token"]])}, params=data)
            five_real_time_pricedata = response.json()

        # 买1

        payload = json.dumps({
            "orders": [
                {
                    "symbol": val_df.CODES.values[0][:6],
                    "exchange": exchangedata,
                    "price": float(five_real_time_pricedata['bjw1']),
                    "quantity": p_values / float(five_real_time_pricedata['bjw1']) // 100 * 100,
                    "operator": "buy"
                }
            ]
        })
        # 委托下单
        response = s.post(f"{beehive3_api}/api/orders/{portfolio_id}/entrust_orders",
                          headers={"Authorization": " ".join(["Token", logined_root_user["token"]])}, data=payload)

    else:
        exchangedata = '1'
        data = {"symbol": val_df.CODES.values[0][:6], "exchange": exchangedata}
        response = s.get(f"{beehive3_api}/api/stock/five_real_time_price",
                         headers={"Authorization": " ".join(["Token", logined_root_user["token"]])}, params=data)
        if response.status_code != 200:
            exchangedata = '0'
            data = {"symbol": val_df.CODES.values[0][:6], "exchange": exchangedata}
            response = s.get(f"{beehive3_api}/api/stock/five_real_time_price",
                             headers={"Authorization": " ".join(["Token", logined_root_user["token"]])}, params=data)
            five_real_time_pricedata = response.json()
        else:
            response = s.get(f"{beehive3_api}/api/stock/five_real_time_price",
                             headers={"Authorization": " ".join(["Token", logined_root_user["token"]])}, params=data)
            five_real_time_pricedata = response.json()

        # 买1
        payload = json.dumps({
            "orders": [
                {
                    "symbol": val_df.CODES.values[0][:6],
                    "exchange": exchangedata,
                    "price": float(five_real_time_pricedata['bjw1']),
                    "quantity": p_values * 0.5 / float(five_real_time_pricedata['bjw1']) // 100 * 100,
                    "operator": "buy"
                }
            ]
        })
        # 委托下单
        response = s.post(f"{beehive3_api}/api/orders/{portfolio_id}/entrust_orders",
                          headers={"Authorization": " ".join(["Token", logined_root_user["token"]])}, data=payload)
        print(val_df.CODES.values[0][:6],"金字塔下单")

# 老板电器
# 片仔癀
# 恒瑞医药
# 我武生物
# 海康威视
# 通策医疗
# 法拉电子
# 海天味业
# 万孚生物
# 47500 = 00000*0.95/10
# 47500/2070.38
# 47500/44.2//100*100
# 47500/428.88//100*100
# 47500/428.88//100*100
# 47500/67.99//100*100
# 47500/67.99//100*100
# 47500/64.60//100*100
# 47500/63.18//100*100
# 47500/392.1//100*100
# 47500/147.43//100*100
# 47500/129.64//100*100
# 47500/62.54//100*100

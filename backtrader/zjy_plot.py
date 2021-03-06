# -*- coding: utf-8 -*-
from collections import OrderedDict
import backtrader as bt
import pandas as pd
import dash_html_components as html

def out_result(cerebro):
    cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')
    cerebro.addanalyzer(bt.analyzers.TotalValue, _name='_TotalValue')
    cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name='_AnnualReturn')
    cerebro.addanalyzer(bt.analyzers.Calmar, _name='_Calmar')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='_DrawDown')
    cerebro.addanalyzer(bt.analyzers.TimeDrawDown, _name='_TimeDrawDown')
    cerebro.addanalyzer(bt.analyzers.GrossLeverage, _name='_GrossLeverage')
    cerebro.addanalyzer(bt.analyzers.PositionsValue, _name='_PositionsValue')
    cerebro.addanalyzer(bt.analyzers.LogReturnsRolling, _name='_LogReturnsRolling')
    cerebro.addanalyzer(bt.analyzers.PeriodStats, _name='_PeriodStats')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='_Returns')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='_SharpeRatio')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio_A, _name='_SharpeRatio_A')
    cerebro.addanalyzer(bt.analyzers.SQN, _name='_SQN')
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='_TimeReturn')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='_TradeAnalyzer')
    cerebro.addanalyzer(bt.analyzers.Transactions, _name='_Transactions')
    cerebro.addanalyzer(bt.analyzers.VWR, _name='_VWR')
    
    results = cerebro.run()
    trade_list = pd.DataFrame(results[0].analyzers.trade_list.get_analysis())
    performance_dict=OrderedDict()
    calmar_ratio=list(results[0].analyzers._Calmar.get_analysis().values())[-1]
    drawdown_info=results[0].analyzers._DrawDown.get_analysis()
    average_drawdown_len=drawdown_info['len']
    average_drawdown_rate=drawdown_info['drawdown']
    average_drawdown_money=drawdown_info['moneydown']
    max_drawdown_len=drawdown_info['max']['len']
    max_drawdown_rate=drawdown_info['max']['drawdown']
    max_drawdown_money=drawdown_info['max']['moneydown']
    PeriodStats_info=results[0].analyzers._PeriodStats.get_analysis()
    average_rate=PeriodStats_info['average']
    stddev_rate=PeriodStats_info['stddev']
    positive_year=PeriodStats_info['positive']
    negative_year=PeriodStats_info['negative']
    nochange_year=PeriodStats_info['nochange']
    best_year=PeriodStats_info['best']
    worst_year=PeriodStats_info['worst']
    SQN_info=results[0].analyzers._SQN.get_analysis()
    sqn_ratio=SQN_info['sqn']
    VWR_info=results[0].analyzers._VWR.get_analysis()
    vwr_ratio=VWR_info['vwr']
    sharpe_info=results[0].analyzers._SharpeRatio.get_analysis()
    # sharpe_info=results[0].analyzers._SharpeRatio_A.get_analysis()
    sharpe_ratio=sharpe_info['sharperatio']

    performance_dict['calmar_ratio']=round(calmar_ratio*100,3)
    performance_dict['average_drawdown_len']=average_drawdown_len
    performance_dict['average_drawdown_rate']=round(average_drawdown_rate,2)
    performance_dict['average_drawdown_money']=round(average_drawdown_money,2)
    performance_dict['max_drawdown_len']=max_drawdown_len
    performance_dict['max_drawdown_rate']=round(max_drawdown_rate,2)
    performance_dict['max_drawdown_money']=round(max_drawdown_money,2)
    performance_dict['average_rate']=round(average_rate,2)
    performance_dict['stddev_rate']=round(stddev_rate,2)
    performance_dict['positive_year']=positive_year
    performance_dict['negative_year']=negative_year
    performance_dict['nochange_year']=nochange_year
    performance_dict['best_year']=round(best_year*100,2)
    performance_dict['worst_year']=round(worst_year*100,2)
    performance_dict['sqn_ratio']=round(sqn_ratio,2)
    performance_dict['vwr_ratio']=round(vwr_ratio,2)
    try:
        performance_dict['sharpe_info']=round(sharpe_ratio,2)
    except:
        performance_dict['sharpe_info'] = 0
    performance_dict['omega']=0

    trade_dict_1=OrderedDict()
    trade_dict_2=OrderedDict()
    trade_info=results[0].analyzers._TradeAnalyzer.get_analysis()
    total_trade_num=trade_info['total']['total']
    total_trade_opened=trade_info['total']['open']
    total_trade_closed=trade_info['total']['closed']
    total_trade_len=trade_info['len']['total']
    long_trade_len=trade_info['len']['long']['total']
    short_trade_len=trade_info['len']['short']['total']
    
    longest_win_num=trade_info['streak']['won']['longest']
    longest_lost_num=trade_info['streak']['lost']['longest']
    net_total_pnl=trade_info['pnl']['net']['total']
    net_average_pnl=trade_info['pnl']['net']['average']
    win_num=trade_info['won']['total']
    win_total_pnl=trade_info['won']['pnl']['total']
    win_average_pnl=trade_info['won']['pnl']['average']
    win_max_pnl=trade_info['won']['pnl']['max']
    lost_num=trade_info['lost']['total']
    lost_total_pnl=trade_info['lost']['pnl']['total']
    lost_average_pnl=trade_info['lost']['pnl']['average']
    lost_max_pnl=trade_info['lost']['pnl']['max']
    
    trade_dict_1['total_trade_num']=total_trade_num
    trade_dict_1['total_trade_opened']=total_trade_opened
    trade_dict_1['total_trade_closed']=total_trade_closed
    trade_dict_1['total_trade_len']=total_trade_len
    trade_dict_1['long_trade_len']=long_trade_len
    trade_dict_1['short_trade_len']=short_trade_len
    trade_dict_1['longest_win_num']=longest_win_num
    trade_dict_1['longest_lost_num']=longest_lost_num
    trade_dict_1['net_total_pnl']=net_total_pnl
    trade_dict_1['net_average_pnl']=net_average_pnl
    trade_dict_1['win_num']=win_num
    trade_dict_1['win_total_pnl']=win_total_pnl
    trade_dict_1['win_average_pnl']=win_average_pnl
    trade_dict_1['win_max_pnl']=win_max_pnl
    trade_dict_1['lost_num']=lost_num
    trade_dict_1['lost_total_pnl']=lost_total_pnl
    trade_dict_1['lost_average_pnl']=lost_average_pnl
    trade_dict_1['lost_max_pnl']=lost_max_pnl
    
    long_num=trade_info['long']['total']
    long_win_num=trade_info['long']['won']
    long_lost_num=trade_info['long']['lost']
    long_total_pnl=trade_info['long']['pnl']['total']
    long_average_pnl=trade_info['long']['pnl']['average']
    long_win_total_pnl=trade_info['long']['pnl']['won']['total']
    long_win_max_pnl=trade_info['long']['pnl']['won']['max']
    long_lost_total_pnl=trade_info['long']['pnl']['lost']['total']
    long_lost_max_pnl=trade_info['long']['pnl']['lost']['max']
    
    short_num=trade_info['short']['total']
    short_win_num=trade_info['short']['won']
    short_lost_num=trade_info['short']['lost']
    short_total_pnl=trade_info['short']['pnl']['total']
    short_average_pnl=trade_info['short']['pnl']['average']
    short_win_total_pnl=trade_info['short']['pnl']['won']['total']
    short_win_max_pnl=trade_info['short']['pnl']['won']['max']
    short_lost_total_pnl=trade_info['short']['pnl']['lost']['total']
    short_lost_max_pnl=trade_info['short']['pnl']['lost']['max']
    
    trade_dict_2['long_num']=long_num
    trade_dict_2['long_win_num']=long_win_num
    trade_dict_2['long_lost_num']=long_lost_num
    trade_dict_2['long_total_pnl']=long_total_pnl
    trade_dict_2['long_average_pnl']=long_average_pnl
    trade_dict_2['long_win_total_pnl']=long_win_total_pnl
    trade_dict_2['long_win_max_pnl']=long_win_max_pnl
    trade_dict_2['long_lost_total_pnl']=long_lost_total_pnl
    trade_dict_2['long_lost_max_pnl']=long_lost_max_pnl
    trade_dict_2['short_num']=short_num
    trade_dict_2['short_win_num']=short_win_num
    trade_dict_2['short_lost_num']=short_lost_num
    trade_dict_2['short_total_pnl']=short_total_pnl
    trade_dict_2['short_average_pnl']=short_average_pnl
    trade_dict_2['short_win_total_pnl']=short_win_total_pnl
    trade_dict_2['short_win_max_pnl']=short_win_max_pnl
    trade_dict_2['short_lost_total_pnl']=short_lost_total_pnl
    trade_dict_2['short_lost_max_pnl']=short_lost_max_pnl

    df00=pd.DataFrame(index=range(len(performance_dict)))
    df01=pd.DataFrame([performance_dict]).T
    df01.columns=['绩效指标值']
    df02=pd.DataFrame([trade_dict_1]).T
    df02.columns=['普通交易指标值']
    df03=pd.DataFrame([trade_dict_2]).T
    df03.columns=['多空交易指标值']
    df00['绩效指标']=df01.index
    df00['绩效指标值']=df01.round(4).values
    df00['普通交易指标']=df02.index
    df00['普通交易指标值']=[round(float(i),4) for i in list(df02['普通交易指标值'])]
    df00['多空交易指标']=df03.index
    df00['多空交易指标值']=[round(float(i),4) for i in list(df03['多空交易指标值'])]
    
    # 账户收益率
    df0=pd.DataFrame([results[0].analyzers._TotalValue.get_analysis()]).T
    df0.columns=['total_value']
    
    # 总的杠杆
    df1=pd.DataFrame([results[0].analyzers._GrossLeverage.get_analysis()]).T
    df1.columns=['GrossLeverage']
    

    # 滚动的对数收益率
    df2=pd.DataFrame([results[0].analyzers._LogReturnsRolling.get_analysis()]).T
    df2.columns=['log_return']
    
    # year_rate
    df3=pd.DataFrame([results[0].analyzers._AnnualReturn.get_analysis()]).T
    df3.columns=['year_rate']
    
    # 总的持仓价值
    df4=pd.DataFrame(results[0].analyzers._PositionsValue.get_analysis()).T
    df4['total_position_value']=df4.sum(axis=1)

    pyfoliozer = results[0].analyzers.getbyname('pyfolio')
    returns, positions, transactions, gross_lev = pyfoliozer.get_pf_items()
    return df00,df0,df1,df2,df3,df4,trade_list

def create_table(df, max_rows=18):
    
    """基于dataframe，设置表格格式"""
    
    table = html.Table(
        # Header
        [
            html.Tr(
                [
                    html.Th(col) for col in df.columns
                ]
            )
        ] +
        # Body
        [
            html.Tr(
                [
                    html.Td(
                        df.iloc[i][col]
                    ) for col in df.columns
                ]
            ) for i in range(min(len(df), max_rows))
        ]   
    )
    return table

# def plot_result(cerebro):
#     df00,df0,df1,df2,df3,df4=out_result(cerebro)
#
#     app = JupyterDash('评价指标')
#     colors = dict(background = 'white', text = 'black')
#     strategy_name="量化策略"
#     app.layout = html.Div(
#         style = dict(backgroundColor = colors['background']),
#         children = [
#
#             html.H1(
#                 children='{}策略评估结果'.format(strategy_name),
#                 style = dict(textAlign='center', color = colors['text'])),
#
#             dcc.Graph(
#                 id='账户价值',
#                 figure = dict(
#                     data = [{'x': list(df0.index), 'y': list(df0.total_value),
#                              #'text':[int(i*1000)/10 for i in list(df3.year_rate)],
#                              'type': 'scatter', 'name': '账户价值',
#                             'textposition':"outside"}],
#                     layout = dict(
#                         title='账户价值',
#                         plot_bgcolor = colors['background'],
#                         paper_bgcolor = colors['background'],
#                         font = dict(color = colors['text'],
#                        )
#                     )
#                 )
#             ),
#
#             dcc.Graph(
#                 id='持仓市值',
#                 figure = dict(
#                     data = [{'x': list(df4.index), 'y': list(df4.total_position_value),
#                              #'text':[int(i*1000)/10 for i in list(df3.year_rate)],
#                              'type': 'scatter', 'name': '持仓市值',
#                             'textposition':"outside"}],
#                     layout = dict(
#                         title='持仓市值',
#                         plot_bgcolor = colors['background'],
#                         paper_bgcolor = colors['background'],
#                         font = dict(color = colors['text']),
#                     )
#                 )
#             ),
#             dcc.Graph(
#                 id='年化收益',
#                 figure = dict(
#                     data = [{'x': list(df3.index), 'y': list(df3.year_rate),
#                              'text':[int(i*1000)/10 for i in list(df3.year_rate)],
#                              'type': 'bar', 'name': '年收益率',
#                             'textposition':"outside"}],
#                     layout = dict(
#                         title='年化收益率',
#                         plot_bgcolor = colors['background'],
#                         paper_bgcolor = colors['background'],
#                         font = dict(color = colors['text']),
#                     )
#                 )
#             ),
#             create_table(df00)
#
#
#         ]
#     )
#
#     return app

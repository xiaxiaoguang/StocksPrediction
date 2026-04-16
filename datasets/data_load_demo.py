import sys

from data_utils import load_data #会显示报错，但是上面加了sys path可以使用
#读取股市数据
date_list, features, labels, stock_list = load_data(
        data_type="ch_daily_processed",
        stock_list="csi500",
        start_date="2010-01-01",
        end_date="2024-12-31",
        contain_bj=False,
        drop_short=False,
        short_len=200,
)

reg_capital = load_data(
        data_type='reg_captital',
        stock_list=stock_list,
    )

#"读取turnover_rate在feature中的列index"
turnover_pos = load_data(
        data_type='feature_col_index', col_name="turnover_rate")
#"读取industry在feature中的列index"
industry_pos = load_data(
        data_type='feature_col_index', col_name="industry")
#"读取validity_label在label中的列index"
vali_label_pos = load_data(
        data_type='label_col_index', col_name="validity_label")

'''
    data_type 读取的数据类型 暂时可读取的有  
        "ch_daily_processed" a股日频数据（处理后）
         'ch_daily_origin' a股日频数据（原始数据）

         'reg_captital' 读取股票注册信息
         'feature_col_index' 读取字段在feature中的列index
         'label_col_index'读取字段在label中的列index


    col_name: 仅读取  'feature_col_index'和 'label_col_index'有效，获取某字段的index
             
    stock_list：股票列表 "all"全部  "csi500"中证500 "csi300"沪深300 "csi800"沪深300+中证500，或自定义list ['000001.SZ','000002.SZ']
    start_date="2010-01-01"起始时间
    end_date="2022-12-31",结束时间
    contain_bj=False,是否包含北交所股票
    drop_short=False,是否去掉交易日过少的股票
    short_len=200, 如果drop_short=True ，则去掉交易日少于short_len的股票

    return:
        若 读取"ch_daily_processed"或'ch_daily_origin' 
            date_list ：区间交易日列表
            features : 特征 包含feature_col
            
            labels : 标签 包含   label_col 

            stock_list  ：所读取的股票列表
若去除交易天数不符股票和北交所股票，可能读取csi800/500/300/all并不是全部股票
'''
#feature字段名称可以从tushare上查询
"""
feature_col = [
        "open",
        "pct_chg",
        "high",
        "low",
        "close",
        "pre_close",
        "ma5",
        "ma10",
        "ma15",
        "ma20",
        "ma25",
        "change",
        "vol",
        "amount",
        "turnover_rate",
        "turnover_rate_f",
        "volume_ratio",
        "pe",
        "pe_ttm",
        "pb",
        "ps",
        "ps_ttm",
        "total_mv",
        "circ_mv",
        "buy_sm_vol",
        "buy_sm_amount",
        "sell_sm_vol",
        "sell_sm_amount",
        "buy_md_vol",
        "buy_md_amount",
        "sell_md_vol",
        "sell_md_amount",
        "buy_lg_vol",
        "buy_lg_amount",
        "sell_lg_vol",
        "sell_lg_amount",
        "buy_elg_vol",
        "buy_elg_amount",
        "sell_elg_vol",
        "sell_elg_amount",
        "net_mf_vol",
        "net_mf_amount",
        "up_limit",
        "down_limit",
        "industry",
    ]

label_col = ["return_10", 未来10天收益率均值
    "return_5", 未来5天收益率均值
    "validity_label", 样本有效性(由于对齐问题进行数据存在补全，标记该条记录是实际数据还是补全数据)
    "pct_chg"   未来一日的收益率
    ]

open: The opening price of the stock for the trading day.
pct_chg: The percentage change in the stock price from the previous closing price.
high: The highest price at which the stock traded during the trading day.
low: The lowest price at which the stock traded during the trading day.
close: The closing price of the stock for the trading day.
pre_close: The closing price of the stock from the previous trading day.
ma5: The 5-day moving average price of the stock.
ma10: The 10-day moving average price of the stock.
ma15: The 15-day moving average price of the stock.
ma20: The 20-day moving average price of the stock.
ma25: The 25-day moving average price of the stock.
change: The absolute change in the stock price from the previous closing price.
vol: The trading volume of the stock for the trading day.
amount: The total monetary value of all shares traded during the trading day.
turnover_rate: The rate at which shares of the stock are traded relative to the total number of available shares.
turnover_rate_f: The floating turnover rate, which only considers the shares that are freely tradable (excluding locked-in shares).
volume_ratio: The ratio of the current trading volume to the average trading volume over a specific period.
pe: The price-to-earnings ratio, calculated as the stock price divided by earnings per share.
pe_ttm: The trailing twelve months price-to-earnings ratio, which uses the earnings of the past 12 months.
pb: The price-to-book ratio, calculated as the stock price divided by book value per share.
ps: The price-to-sales ratio, calculated as the stock price divided by sales per share.
ps_ttm: The trailing twelve months price-to-sales ratio, using sales of the past 12 months.
total_mv: The total market value of the company, calculated as the current stock price multiplied by the total number of shares outstanding.
circ_mv: The circulating market value, which is the market value of the shares that are freely tradable.
buy_sm_vol: The volume of small trades (usually by retail investors) where shares were bought.
buy_sm_amount: The total monetary value of small trades where shares were bought.
sell_sm_vol: The volume of small trades where shares were sold.
sell_sm_amount: The total monetary value of small trades where shares were sold.
buy_md_vol: The volume of medium trades (a size between small and large) where shares were bought.
buy_md_amount: The total monetary value of medium trades where shares were bought.
sell_md_vol: The volume of medium trades where shares were sold.
sell_md_amount: The total monetary value of medium trades where shares were sold.
buy_lg_vol: The volume of large trades (usually by institutional investors) where shares were bought.
buy_lg_amount: The total monetary value of large trades where shares were bought.
sell_lg_vol: The volume of large trades where shares were sold.
sell_lg_amount: The total monetary value of large trades where shares were sold.
buy_elg_vol: The volume of extra-large trades where shares were bought.
buy_elg_amount: The total monetary value of extra-large trades where shares were bought.
sell_elg_vol: The volume of extra-large trades where shares were sold.
sell_elg_amount: The total monetary value of extra-large trades where shares were sold.
net_mf_vol: The net volume of money flow, calculated as the volume of shares bought minus the volume of shares sold.
net_mf_amount: The net amount of money flow, calculated as the monetary value of shares bought minus the monetary value of shares sold.
up_limit: The upper price limit for the stock during the trading day.
down_limit: The lower price limit for the stock during the trading day.
industry: The industry sector to which the stock belongs.    
"""
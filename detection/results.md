完了，因为异常失败数据引入数据集有变动了，又需要重跑

## iTransformer

METRIC                 | 55% ORACLE (ALL MIN) | 55% ORACLE (ANOMALY)
Cumulative Return      |           21.61% |           118.71%
Annualized Sharpe      |             1.33 |              3.64
Maximum Drawdown       |           -0.14% |            -0.11%
Time in Market (Exp.)  |           99.58% |            99.52%
Avg Stocks Held/Day    |             39.1 |               7.9
test_Accuracy: 0.1147, test_F1: 0.2058, test_AUC: 0.6013, test_Local_Accuracy: 0.8354, test_Local_Precision: 0.0401, test_Local_Recall: 0.4476, test_Local_F1: 0.0736, test_Local_AUC: 0.7108

-----------------------------------------------------------------
METRIC                 | 55% ORACLE (ALL MIN) | 55% ORACLE (ANOMALY)
Cumulative Return      |           21.61% |            33.22%
Annualized Sharpe      |             1.33 |              1.92
Maximum Drawdown       |           -0.11% |            -0.08%
Time in Market (Exp.)  |           99.58% |            99.58%
Avg Stocks Held/Day    |             39.1 |              35.5
-----------------------------------------------------------------

test_Accuracy: 0.6322, test_F1: 0.4413, test_AUC: 0.7261, test_Local_Accuracy: 0.1560, test_Local_Precision: 0.0650, test_Local_Recall: 0.9814, test_Local_F1: 0.1220, test_Local_AUC: 0.7340


## iMamba

METRIC                 | 55% ORACLE (ALL MIN) | 55% ORACLE (ANOMALY)
Cumulative Return      |           21.61% |           403.70%
Annualized Sharpe      |             1.33 |              7.39
Maximum Drawdown       |           -0.14% |            -0.03%
Time in Market (Exp.)  |           99.58% |            99.58%
Avg Stocks Held/Day    |             39.1 |               9.2
test_Accuracy: 0.2981, test_F1: 0.2386, test_AUC: 0.7400, test_Local_Accuracy: 0.7819, test_Local_Precision: 0.0467, test_Local_Recall: 0.7164, test_Local_F1: 0.0876, test_Local_AUC: 0.8267


-----------------------------------------------------------------
METRIC                 | 55% ORACLE (ALL MIN) | 55% ORACLE (ANOMALY)
Cumulative Return      |           21.61% |           154.96%
Annualized Sharpe      |             1.33 |              5.24
Maximum Drawdown       |           -0.11% |            -0.03%
Time in Market (Exp.)  |           99.58% |            99.58%
Avg Stocks Held/Day    |             39.1 |              21.4
-----------------------------------------------------------------
test_Accuracy: 0.6568, test_F1: 0.5887, test_AUC: 0.7183, test_Local_Accuracy: 0.5061, test_Local_Precision: 0.0973, test_Local_Recall: 0.8783, test_Local_F1: 0.1752, test_Local_AUC: 0.7729


## NumerMOE

旧指标，3倍

METRIC                 | 55% ORACLE (ALL MIN) | 55% ORACLE (ANOMALY)
Cumulative Return      |           21.61% |           628.01%
Annualized Sharpe      |             1.33 |              9.86
Maximum Drawdown       |           -0.14% |            -0.01%
Time in Market (Exp.)  |           99.58% |            40.87%
Avg Stocks Held/Day    |             39.1 |               1.0
test_Accuracy: 0.8853, test_F1: 0.0000, test_AUC: 0.4554, test_Local_Accuracy: 0.9714, test_Local_Precision: 0.1166, test_Local_Recall: 0.1452, test_Local_F1: 0.1294, test_Local_AUC: 0.6160

新指标，2倍

-----------------------------------------------------------------
METRIC                 | 55% ORACLE (ALL MIN) | 55% ORACLE (ANOMALY)
Cumulative Return      |           21.61% |            26.33%
Annualized Sharpe      |             1.33 |              1.56
Maximum Drawdown       |           -0.11% |            -0.10%
Time in Market (Exp.)  |           99.58% |            99.58%
Avg Stocks Held/Day    |             39.1 |              37.3
-----------------------------------------------------------------

test_Accuracy: 0.4672, test_F1: 0.6367, test_AUC: 0.4561, test_Local_Accuracy: 0.1035, test_Local_Precision: 0.0613, test_Local_Recall: 0.9793, test_Local_F1: 0.1154, test_Local_AUC: 0.6238

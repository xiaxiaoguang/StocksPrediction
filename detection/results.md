

五分层32*5 + 4layers + 4expert/top2

2026-05-06 14:07:15,488 - easytorch-training - INFO - -----------------------------------------------------------------
2026-05-06 14:07:15,488 - easytorch-training - INFO - METRIC                 | 55% ORACLE (ALL MIN) | 55% ORACLE (ANOMALY)
2026-05-06 14:07:15,489 - easytorch-training - INFO - Cumulative Return      |           22.04% |           511.27%
2026-05-06 14:07:15,489 - easytorch-training - INFO - Annualized Sharpe      |             1.31 |              8.76
2026-05-06 14:07:15,489 - easytorch-training - INFO - Maximum Drawdown       |           -0.16% |            -0.01%
2026-05-06 14:07:15,489 - easytorch-training - INFO - Time in Market (Exp.)  |           99.58% |            99.58%
2026-05-06 14:07:15,489 - easytorch-training - INFO - Avg Stocks Held/Day    |             39.2 |              12.1
2026-05-06 14:07:15,489 - easytorch-training - INFO - -----------------------------------------------------------------

2026-05-06 14:07:15,491 - easytorch-training - INFO - Result <test>: [test_time: 10.18 (s), test_Accuracy: 0.9703, test_F1: 0.0000, test_AUC: 0.6048, test_Local_Accuracy: 0.7200, test_Local_Precision: 0.2201, test_Local_Recall: 0.7134, test_Local_F1: 0.3364, test_Local_AUC: 0.7836]


四分层64*4，最后加norm，1，2，4，8

2026-05-06 10:24:12,791 - easytorch-training - INFO - -----------------------------------------------------------------
2026-05-06 10:24:12,791 - easytorch-training - INFO - METRIC                 | 55% ORACLE (ALL MIN) | 55% ORACLE (ANOMALY)
2026-05-06 10:24:12,791 - easytorch-training - INFO - Cumulative Return      |           22.04% |           523.06%
2026-05-06 10:24:12,791 - easytorch-training - INFO - Annualized Sharpe      |             1.31 |              8.48
2026-05-06 10:24:12,791 - easytorch-training - INFO - Maximum Drawdown       |           -0.16% |            -0.01%
2026-05-06 10:24:12,791 - easytorch-training - INFO - Time in Market (Exp.)  |           99.58% |            99.58%
2026-05-06 10:24:12,791 - easytorch-training - INFO - Avg Stocks Held/Day    |             39.2 |              10.6
2026-05-06 10:24:12,791 - easytorch-training - INFO - -----------------------------------------------------------------

2026-05-06 10:24:12,793 - easytorch-training - INFO - Result <test>: [test_time: 9.98 (s), test_Accuracy: 0.0297, test_F1: 0.0577, test_AUC: 0.4667, test_Local_Accuracy: 0.7597, test_Local_Precision: 0.2352, test_Local_Recall: 0.6285, test_Local_F1: 0.3423, test_Local_AUC: 0.7754]

把final_fusion加上，然后五分层，效果就不好了

2026-05-06 12:38:32,776 - easytorch-training - INFO - -----------------------------------------------------------------
2026-05-06 12:38:32,776 - easytorch-training - INFO - METRIC                 | 55% ORACLE (ALL MIN) | 55% ORACLE (ANOMALY)
2026-05-06 12:38:32,776 - easytorch-training - INFO - Cumulative Return      |           22.04% |           401.35%
2026-05-06 12:38:32,777 - easytorch-training - INFO - Annualized Sharpe      |             1.31 |              7.79
2026-05-06 12:38:32,777 - easytorch-training - INFO - Maximum Drawdown       |           -0.16% |            -0.01%
2026-05-06 12:38:32,777 - easytorch-training - INFO - Time in Market (Exp.)  |           99.58% |            99.58%
2026-05-06 12:38:32,777 - easytorch-training - INFO - Avg Stocks Held/Day    |             39.2 |              13.7
2026-05-06 12:38:32,777 - easytorch-training - INFO - -----------------------------------------------------------------
2026-05-06 12:38:32,778 - easytorch-training - INFO - Result <test>: [test_time: 13.77 (s), test_Accuracy: 0.4644, test_F1: 0.0656, test_AUC: 0.5745, test_Local_Accuracy: 0.6971, test_Local_Precision: 0.2104, test_Local_Recall: 0.7423, test_Local_F1: 0.3279, test_Local_AUC: 0.7790]

这个是五分层然后d_ff调大到512，嵌入纬度是320，应该是不能随便调大d_ff

2026-05-06 14:38:26,120 - easytorch-training - INFO - -----------------------------------------------------------------
2026-05-06 14:38:26,120 - easytorch-training - INFO - METRIC                 | 55% ORACLE (ALL MIN) | 55% ORACLE (ANOMALY)
2026-05-06 14:38:26,120 - easytorch-training - INFO - Cumulative Return      |           22.04% |           236.66%
2026-05-06 14:38:26,120 - easytorch-training - INFO - Annualized Sharpe      |             1.31 |              6.50
2026-05-06 14:38:26,120 - easytorch-training - INFO - Maximum Drawdown       |           -0.16% |            -0.02%
2026-05-06 14:38:26,120 - easytorch-training - INFO - Time in Market (Exp.)  |           99.58% |            99.58%
2026-05-06 14:38:26,120 - easytorch-training - INFO - Avg Stocks Held/Day    |             39.2 |              15.7
2026-05-06 14:38:26,120 - easytorch-training - INFO - -----------------------------------------------------------------
2026-05-06 14:38:26,122 - easytorch-training - INFO - Result <test>: [test_time: 14.56 (s), test_Accuracy: 0.0297, test_F1: 0.0577, test_AUC: 0.4459, test_Local_Accuracy: 0.6407, test_Local_Precision: 0.1795, test_Local_Recall: 0.7310, test_Local_F1: 0.2882, test_Local_AUC: 0.6999]


这个也不行，timeencoder消融到只有1，4，16三维，效果也不好

-----------------------------------------------------------------
2026-05-06 15:48:45,715 - easytorch-training - INFO - METRIC                 | 55% ORACLE (ALL MIN) | 55% ORACLE (ANOMALY)
2026-05-06 15:48:45,715 - easytorch-training - INFO - Cumulative Return      |           22.04% |           478.11%
2026-05-06 15:48:45,716 - easytorch-training - INFO - Annualized Sharpe      |             1.31 |              8.56
2026-05-06 15:48:45,716 - easytorch-training - INFO - Maximum Drawdown       |           -0.16% |            -0.01%
2026-05-06 15:48:45,716 - easytorch-training - INFO - Time in Market (Exp.)  |           99.58% |            99.58%
2026-05-06 15:48:45,716 - easytorch-training - INFO - Avg Stocks Held/Day    |             39.2 |              12.4
2026-05-06 15:48:45,716 - easytorch-training - INFO - -----------------------------------------------------------------
2026-05-06 15:48:45,717 - easytorch-training - INFO - Result <test>: [test_time: 10.65 (s), test_Accuracy: 0.0297, test_F1: 0.0577, test_AUC: 0.4256, test_Local_Accuracy: 0.7142, test_Local_Precision: 0.2165, test_Local_Recall: 0.7153, test_Local_F1: 0.3324, test_Local_AUC: 0.7816]

只使用1，2，4，三维，效果稍微差一点，还是1，2，4，8效果更好

2026-05-06 16:40:06,684 - easytorch-training - INFO - -----------------------------------------------------------------
2026-05-06 16:40:06,684 - easytorch-training - INFO - METRIC                 | 55% ORACLE (ALL MIN) | 55% ORACLE (ANOMALY)
2026-05-06 16:40:06,684 - easytorch-training - INFO - Cumulative Return      |           22.04% |           534.72%
2026-05-06 16:40:06,684 - easytorch-training - INFO - Annualized Sharpe      |             1.31 |              8.26
2026-05-06 16:40:06,684 - easytorch-training - INFO - Maximum Drawdown       |           -0.16% |            -0.01%
2026-05-06 16:40:06,684 - easytorch-training - INFO - Time in Market (Exp.)  |           99.58% |            99.58%
2026-05-06 16:40:06,684 - easytorch-training - INFO - Avg Stocks Held/Day    |             39.2 |              10.4
2026-05-06 16:40:06,684 - easytorch-training - INFO - -----------------------------------------------------------------
2026-05-06 16:40:06,686 - easytorch-training - INFO - Result <test>: [test_time: 8.71 (s), test_Accuracy: 0.0297, test_F1: 0.0577, test_AUC: 0.4810, test_Local_Accuracy: 0.7603, test_Local_Precision: 0.2347, test_Local_Recall: 0.6234, test_Local_F1: 0.3410, test_Local_AUC: 0.7746]




e58d0b00b2361dd9fb595093d58dec89
移除learnable spatial embedding在最优设置【四分层64*4，最后加norm，1，2，4，8】的基础上

效果很差不用看了

#-*-coding:utf-8 -*-

# Covid-19疫情的SEIR模型的预测


import matplotlib.pyplot as plt
import pandas as pd
from math import *
import datetime
import matplotlib.dates as mdates
from src.Dynamic_SEIR_model import *
from src.helper_fun_epi_model import *

import os
import warnings

from matplotlib.pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']
warnings.filterwarnings('ignore')


## 部分假设
China_population = 1400000000


### 加载和清洗数据


## 加载数据
df = pd.read_csv("data/DXYArea.csv")
"""
数据清洗 
"""
df['date'] = pd.to_datetime(df['date'])
df = df[df['date'] > datetime.datetime(2019, 12, 7)]  # 选2019-12-08，即第一例患者出现日期
# df = df[df['date'] > datetime.datetime(2020, 1, 22)] # 开始日期为2020-1-23, 在武汉隔离之后————————有意义吗？既然采取了隔离措施，其传染率必然会下降
df = df[df['date'] != df['date'].max()] # 因为今天（运行程序当天）的记录不全，故移除
## 数据准备
df['R'] = df['cured'] + df['dead']
SEIR_data = df[['date', 'Days', 'countryCode', 'province', 'city', 'net_confirmed', 'suspected', 'R',
              ]].rename(columns={"net_confirmed": "I", "suspected": "E"})
# print("Information of SEIR_data: \n")
# print(SEIR_data.info())


# 使用2020年2月14日前的数据作为训练集
df = df[df['date'] != df['date'].max()] # 因为今天（运行程序当天）的记录不全，故移除
df['R'] = df['cured'] + df['dead']
SEIR_data = df[['date', 'Days', 'countryCode','province', 'city', 'net_confirmed', 'suspected', 'R',
              ]].rename(columns={"net_confirmed": "I", "suspected": "E"})
# print(SEIR_data.info())
China_df = SEIR_data[SEIR_data['date'] < datetime.datetime(2020, 2, 14)]
China_total = get_China_total(China_df)


# 全国
###############################################################################
df_1 = pd.read_csv("data/DXYArea.csv")
df_1['date'] = pd.to_datetime(df_1['date'])
# df_1 = df_1[df_1['date'] > datetime.datetime(2019, 12, 7)]  # 选2019-12-08，即第一例患者出现日期
df_1 = df_1[df_1['date'] > datetime.datetime(2020, 1, 22)] # first day is 2020-1-23, 在武汉隔离之后————————有意义吗？既然采取了隔离措施，其传染率必然会下降
df_1 = df_1[df_1['date'] != df_1['date'].max()] # 因为今天（运行程序当天）的记录不全，故移除
## 数据准备
df_1['R'] = df_1['cured'] + df_1['dead']
SEIR_data_1 = df_1[['date', 'Days', 'countryCode','province', 'city', 'net_confirmed', 'suspected', 'R',
              ]].rename(columns={"net_confirmed": "I", "suspected": "E"})
df_1 = df_1[df_1['date'] != df_1['date'].max()]
df_1['R'] = df_1['cured'] + df_1['dead']
SEIR_data_1 = df_1[['date', 'Days', 'countryCode','province', 'city', 'net_confirmed', 'suspected', 'R',
              ]].rename(columns={"net_confirmed": "I", "suspected": "E"})
China_df_1 = SEIR_data_1[SEIR_data_1['date'] < datetime.datetime(2020, 2, 14)]
China_total_1 = get_China_total(China_df_1)


### 估计国内总数

 
#----------------------------------------------epoch从10000改为10000------------------------------------------
# -----------------------------------Dynamic_SEIR_1
Dynamic_SEIR_1 = Train_Dynamic_SEIR(epoch = 10000, data = China_total,
                 population = 1400000000, rateEI = 1/7, rateIR=1/14, c = 1, b = -10, alpha = 0.08)

estimation_df = Dynamic_SEIR_1.train()

est_beta_1 = Dynamic_SEIR_1.rateSE
est_alpha_1 = Dynamic_SEIR_1.alpha
est_b_1 = Dynamic_SEIR_1.b
est_c_1 = Dynamic_SEIR_1.c
population = Dynamic_SEIR_1.numIndividuals
Dynamic_SEIR_1.plot_fitted_beta_R0(China_total)

# -----------------------------------Dynamic_SEIR_2
Dynamic_SEIR_2 = Train_Dynamic_SEIR(epoch = 10000, data = China_total_1,
                 population = 1400000000, rateEI = 1/7, rateIR=1/14, c = 1, b = -10, alpha = 0.08)

estimation_df = Dynamic_SEIR_2.train()

est_beta = Dynamic_SEIR_2.rateSE
est_alpha = Dynamic_SEIR_2.alpha
est_b = Dynamic_SEIR_2.b
est_c = Dynamic_SEIR_2.c
population = Dynamic_SEIR_2.numIndividuals
Dynamic_SEIR_2.plot_fitted_result(China_total_1)

# -----------------------------------Dynamic_SEIR_3
I0 = list(China_total['I'])[-1]
R0 = list(China_total['R'])[-1]
E0 = list(China_total['E'])[-1]
S0 = population - I0 - E0 - E0

seir_new = dynamic_SEIR(eons=59, Susceptible=S0, Exposed=E0,
                        Infected=I0, Resistant=R0, rateIR=1 / 14,
                        rateEI=1 / 7, alpha=est_alpha_1, c=est_c_1, b=est_b_1,
                        past_days=China_total['Days'].max()
                        # past_days=59
                        )
# result = seir_new.run(death_rate=0.3)
result = seir_new.run(death_rate=0.2) # 假设死亡率为2%

seir_new.plot_noSuscep('使用SEIR模型预测国内疫情', '人数', '日期', starting_point=China_total['date'].max())





## 将最后的观测值(Observation)用作新SEIR模型的初始点

# I是确诊病例 (total confirmed case - heal - died)
I0 = list(China_total['I'])[-1]
R0 = list(China_total['R'])[-1]
# -------------------------------------------------将4倍修改为3倍
# 假设潜伏期内的个体总数是当前易感病例的4倍
E0 = list(China_total['E'])[-1] * 3
S0 = population - I0 - E0 - R0

# --------------------------------------------------------------------------------------------------------
# eons——建模时间点数量（预测天数），原为59，现改为29
seir_new = dynamic_SEIR(eons=64, Susceptible=S0, Exposed = E0,
                    Infected=I0, Resistant=R0, rateIR=1/14,
                    rateEI = 1/7, alpha = est_alpha, c = est_c, b = est_b,
                    past_days = China_total['Days'].max())
result = seir_new.run(death_rate = 0.02) # 假设死亡率为2％

seir_new.plot_noSuscep('SEIR模型预测全国感染人数', '人数', '日期',
                       starting_point = China_total['date'].max())


 

'''
使用SEIR模型结果计算MAPE测试分数
'''
test = get_China_total(SEIR_data[SEIR_data['date'] >= datetime.datetime(2020, 2, 14)])


plot_test_data_with_MAPE_Infected(test, result, title="确诊人数变化")

plot_test_data_with_MAPE_Resistant(test, result, title='治愈/死亡人数变化')
plt.show()


#-*-coding:GBK -*-

import pandas as pd
import numpy as np
import operator
import matplotlib.pyplot as plt
import pandas as pd
import pandas
from math import *
import datetime
import matplotlib.dates as mdates
# from helper_fun_epi_model import *
from src.helper_fun_epi_model import *
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
import warnings

warnings.filterwarnings('ignore')
'''
from matplotlib.font_manager import FontProperties
font_set = FontProperties(fname='SimHei.ttf')
'''
# 修改标题为中文
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']

class dynamic_SEIR:
    """
    'eons' (number of time points to model, default 1000) 建模时间点的数量

    'Susceptible' (number of susceptible individuals at time 0, default 950) 0时刻易感人数
    'Exposed' (number of individuals during incubation period) 潜伏期人数
    'Infected' (number of infected individuals at time 0, default 50) 0时刻感染人数
    'Resistant' (number of resistant individuals at time 0, default 0) 0时刻治愈/死亡人数

    'rateSE'  S到E的转化率，默认0.05
    'rateEI'  E到I的转化率，默认0.1
    'rateIR'  I到R转化率，默认0.01
    即绘制rateSE、rateEI、rateIR三个的变化曲线
    """

    def __init__(self, eons=1000, Susceptible=950, Exposed=100, Infected=50, Resistant=0, rateIR=0.01, rateEI=0.1,
                 alpha=0.3, c=5, b=-10, past_days=30):
        self.eons = eons  # 预测天数
        self.Susceptible = Susceptible
        self.Exposed = Exposed
        self.Infected = Infected
        self.Resistant = Resistant
        self.rateSE = None
        self.rateIR = rateIR
        self.rateEI = rateEI
        self.numIndividuals = Susceptible + Infected + Resistant + Exposed  # S、E、I、R的总人数
        self.alpha = alpha
        self.c = c
        self.b = b
        self.past_days = past_days  # 自上次观察以来做出预测
        self.results = None
        self.modelRun = False

    def _calculate_beta(self, c: float, t: int, alpha: float, b: float, past_days: int):
        """
        根据某些函数计算beta
        """
        t = t + past_days
        return c * exp(-alpha * (t + b)) * pow((1 + exp(-alpha * (t + b))), -2)

    def run(self, death_rate):
        Susceptible = [self.Susceptible]
        Exposed = [self.Exposed]
        Infected = [self.Infected]
        Resistant = [self.Resistant]

        for i in range(1, self.eons):  # 预测的天数
            self.rateSE = self._calculate_beta(c=self.c, t=i, b=self.b,
                                               alpha=self.alpha, past_days=self.past_days)

            S_to_E = (self.rateSE * Susceptible[-1] * Infected[-1]) / self.numIndividuals
            E_to_I = (self.rateEI * Exposed[-1])
            I_to_R = (Infected[-1] * self.rateIR)

            Susceptible.append(Susceptible[-1] - S_to_E)
            Exposed.append(Exposed[-1] + S_to_E - E_to_I)
            Infected.append(Infected[-1] + E_to_I - I_to_R)
            Resistant.append(Resistant[-1] + I_to_R)


        # 死亡数 = 死亡率 * 恢复人数(recovery group)
        Death = list(map(lambda x: (x * death_rate), Resistant))

        # 治愈数 = 移除数(removed) - 死亡数
        Heal = list(map(lambda x: (x * (1 - death_rate)), Resistant))
        self.results = pd.DataFrame.from_dict({'Time': list(range(len(Susceptible))),
                                               'Susceptible': Susceptible, 'Exposed': Exposed, 'Infected': Infected,
                                               'Resistant': Resistant,
                                               'Death': Death, 'Heal': Heal},
                                              orient='index').transpose()
        self.modelRun = True
        return self.results

    def plot(self, title, ylabel, xlabel, starting_point):
        if self.modelRun == False:
            print('Error: Model has not run')
            return
        print("Maximum infected case: ",
              format(int(max(self.results['Infected']))))
        fig, ax = plt.subplots(figsize=(15, 6))
        plt.plot(self.results['Time'], self.results['Susceptible'], linewidth=3,color='blue')
        plt.plot(self.results['Time'], self.results['Infected'],linewidth=3, color='red')
        plt.plot(self.results['Time'], self.results['Exposed'], linewidth=3,color='orange')
        plt.plot(self.results['Time'], self.results['Resistant'], linewidth=3,color='palegreen')
        plt.plot(self.results['Time'], self.results['Heal'],linewidth=3, color='green')
        plt.plot(self.results['Time'], self.results['Death'], linewidth=3,color='grey')
        # set x trick
        datemin = starting_point
        numdays = len(self.results)
        labels = list((datemin + datetime.timedelta(days=x)).strftime('%m-%d') for x in range(numdays))
        plt.xticks(list(range(numdays)), labels, rotation=60)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        plt.legend(['易感人数', '感染人数', '潜伏人数', '移除人数', '治愈人数', '死亡人数'], prop={'size': 12},
                   ncol=6, fancybox=True, shadow=True, loc=1)

        plt.title(title, fontsize=20)
        plt.show()


    def plot_noSuscep(self, title, ylabel, xlabel, starting_point):
        fig, ax = plt.subplots(figsize=(15, 6))
        plt.plot(self.results['Time'], self.results['Infected'], linewidth=3, color='red')
        plt.plot(self.results['Time'], self.results['Resistant'], linewidth=3,color='palegreen')
        plt.plot(self.results['Time'], self.results['Exposed'], linewidth=3,color='orange')
        plt.plot(self.results['Time'], self.results['Heal'], linewidth=3,color='green')
        plt.plot(self.results['Time'], self.results['Death'], linewidth=3, color='grey')
        # 设置 x trick
        datemin = starting_point
        numdays = len(self.results)
        labels = list((datemin + datetime.timedelta(days=x)).strftime('%m-%d') for x in range(numdays))
        plt.xticks(list(range(numdays)), labels, rotation=60, fontsize=18)
        plt.yticks(fontsize=18)
        plt.xlabel(xlabel, fontsize=22)
        plt.ylabel(ylabel, fontsize=20)
        # plt.legend(['Infected', 'Removed', 'Exposed', 'Heal', 'Death'], prop={'size': 12},
        #            ncol=5, fancybox=True, shadow=True, loc='upper left')

        plt.legend(['感染', '移除', '潜伏', '治愈', '死亡'], prop={'size': 14},
                              ncol=5, fancybox=True, shadow=True, loc='upper left')

        plt.title(title, fontsize=25)
        plt.show()



class Train_Dynamic_SEIR:
    """
    'eons' (number of time points to model, default 1000)
    'Susceptible' (number of susceptible individuals at time 0, default 950)  0时刻易感染人数，默认为950
    'Exposed' (number of individuals during incubation period)          0时刻潜伏期的人数
    'Infected' (number of infected individuals at time 0, default 50)   0时刻已感染的人数，默认为50
    'Resistant' (number of resistant individuals at time 0, default 0)  0时刻治愈/死亡的人数，默认为0
    'rateSE' (base rate 'beta' from S to E, default 0.05)   从S到E的beta值
    'rateIR' (base rate 'gamma' from I to R, default 0.01)  从I到R的gamma值
    'rateEI' (base rate of isolation 'altha', from E to I, default 0.1)  从E到I的alpha值
    """

    def __init__(self, data: pandas.core.frame.DataFrame,
                 population: int, epoch=1000, rateIR=0.01, rateEI=0.1, c=1, b=-3, alpha=0.1):
        self.epoch = epoch  # change weights in each epoch
        self.steps = len(data)
        # 实际观测
        self.Exposed = list(data['E'])
        self.Infected = list(data['I'])
        self.Resistant = list(data['R'])
        self.Susceptible = list(population - data['E'] - data['I'] - data['R'])
        # estimation
        self.S_pre = []
        self.E_pre = []
        self.I_pre = []
        self.R_pre = []

        self.rate_S_to_E = []
        self.rate_E_to_I = []
        self.rate_I_to_R = []


        self.past_days = data['Days'].min()  # count the number of days before the first traning point

        # SEIR模型中其他参数
        self.c = c  # 初步猜测
        self.b = b  # 初步猜测
        self.alpha = alpha  # 初步猜测
        self.rateSE = self._calculate_beta(c=self.c, t=0, b=self.b, alpha=self.alpha)  # 初步猜测
        self.rateIR = rateIR
        self.rateEI = rateEI
        self.numIndividuals = population  # 总人数
        self.results = None
        self.estimation = None
        self.modelRun = False
        self.loss = None
        self.betalist = []

    """
    计算beta值
    """
    def _calculate_beta(self, c: float, t: int, alpha: float, b: float):

        return c * exp(-alpha * (t + b)) * pow((1 + exp(-alpha * (t + b))), -2)

    def _calculate_loss(self):
        """
        loss = 所有loss平方之和的开方
        """
        return mean_squared_error(self.Infected, self.I_pre)

    def _calculate_MAPE(self):
        """
        计算模拟(estimated)值与拟合(fitted)值之间的MAPE
        """
        y = np.array(self.Infected)
        y_pred = np.array(self.I_pre)
        mape = np.abs((y - y_pred)) / np.abs(y)
        return np.mean(mape)

    def _update(self):
        """
        训练函数train()的帮助函数（即hepler function)
        在一次迭代中使用梯度下降，试找出全局最小参数
        计算新梯度并更新参数
        """
        E = 2.71828182846
        alpha_eta = 0.000000000000001  # 学习率
        b_eta = 0.00000000001  # 学习率
        c_eta = 0.0000000000001  # 学习率
        alpha_temp = 0.0
        c_temp = 0.0
        b_temp = 0.0
        for t in range(0, self.steps):
            formula = E ** (self.alpha * (t + self.b))
            formula2 = E ** (-self.alpha * (t + self.b))

            loss_to_beta = -2 * (self.Infected[t] - self.I_pre[t]) * (self.I_pre[t]) * t * self.Susceptible[
                t] / self.numIndividuals

            # 使用链规则计算偏导数
            beta_to_alpha = -self.c * formula * (t + self.b) * (formula - 1) * pow((1 + formula), -3)
            beta_to_b = -self.c * formula * self.alpha * (formula - 1) * pow((1 + formula), -3)
            beta_to_c = formula2 * pow((1 + formula2), -2)

            alpha_temp += loss_to_beta * beta_to_alpha  # 新梯度
            b_temp += loss_to_beta * beta_to_b  # 新梯度
            c_temp += loss_to_beta * beta_to_c  # 新梯度

        self.alpha -= alpha_eta * alpha_temp  # 更新alpha、b和c
        self.b -= b_eta * b_temp
        self.c -= c_eta * c_temp
        # print('alpha、b、c的值如下: ')
        # print("c: {}, b: {}, alpha: {}".format(self.c, self.b, self.alpha))

    def train(self):
        """
        使用实时数据进入SEIR模型进行估算
        通过epoch的迭代改善估计参数
        目标:
            使用梯度下降通过最小损失函数找到最佳beta（接触率）

        梯度下降:
                为了解决梯度，我们使用新的alpha，c和b值迭代数据点，计算偏导数
                新的梯度告诉我们成本函数（cost function）在当前位置（当前参数值）的斜率和我们应该更新参数的方向
                我们更新的大小由学习率控制。 （请参见上面的_update（）函数）
        """
        for e in range(self.epoch):
            # 预测列表
            self.S_pre = []
            self.E_pre = []
            self.I_pre = []
            self.R_pre = []

            self.rate_S_to_E = []
            self.rate_E_to_I = []
            self.rate_I_to_R = []


            # 逐步进行预测
            for t in range(0, self.steps):
                if t == 0:
                    self.S_pre.append(self.Susceptible[0])
                    self.E_pre.append(self.Exposed[0])
                    self.I_pre.append(self.Infected[0])
                    self.R_pre.append(self.Resistant[0])
                    self.rateSE = self._calculate_beta(c=self.c, t=t, b=self.b,
                                                       alpha=self.alpha)
                    # print("time {}, beta {}".format(t, self.rateSE))

                    # 存储最佳拟合的Beta
                    if e == (self.epoch - 1):
                        self.betalist.append(self.rateSE)

                else:
                    self.rateSE = self._calculate_beta(c=self.c, t=t, b=self.b,
                                                       alpha=self.alpha)
                    # print("time {}, beta {}".format(t, self.rateSE))

                    # 收集最佳拟合的Beta
                    if e == (self.epoch - 1):
                        self.betalist.append(self.rateSE)

                    # 将实时数据应用于SEIR模型
                    S_to_E = (self.rateSE * self.Susceptible[t] * self.Infected[t]) / self.numIndividuals
                    E_to_I = (self.rateEI * self.Exposed[t])
                    I_to_R = (self.Infected[t] * self.rateIR)
                    self.S_pre.append(self.Susceptible[t] - S_to_E)
                    self.E_pre.append(self.Exposed[t] + S_to_E - E_to_I)
                    self.I_pre.append(self.Infected[t] + E_to_I - I_to_R)
                    self.R_pre.append(self.Resistant[t] + I_to_R)


            # 记录最后一次迭代时的估计值
            if e == (self.epoch - 1):
                self.estimation = pd.DataFrame.from_dict({'Time': list(range(len(self.Susceptible))),
                                                          'Estimated_Susceptible': self.S_pre,
                                                          'Estimated_Exposed': self.E_pre,
                                                          'Estimated_Infected': self.I_pre,
                                                          'Estimated_Resistant': self.R_pre,
                                                          },
                                                         orient='index').transpose()
                self.loss = self._calculate_loss()
                MAPE = self._calculate_MAPE()
                print("The loss in is {}".format(self.loss))
                # print("The MAPE in the whole period is {}".format(MAPE))
                # print("Optimial beta(contact rate) is {}".format(self.rateSE))

                print('全局MAPE值为{}'.format(MAPE))
                print('最佳接触率为{}'.format(self.rateSE))

            # 计算每次迭代中的损失
            self.loss = self._calculate_loss()
            # print("The loss in iteration {} is {}".format(e, self.loss))
            # print("Current beta is {}".format(self.rateSE))

            # 类机器学习的优化方法
            self._update()  # 在每个步骤中使用“梯度下降”更新参数

        return self.estimation  # 最新的估计值

    def plot_fitted_beta_R0(self, real_obs: pandas.core.frame.DataFrame):
        fig, ax = plt.subplots(figsize=(17, 6))
        plt.plot(self.estimation['Time'],  self.betalist, linewidth=3,color='firebrick')# 绘制接触率
        Rlist = [x / self.rateIR for x in self.betalist]  # transmissibility over time
        plt.plot(self.estimation['Time'], Rlist, linewidth=3,color='steelblue')

        # set x tricks
        datemin = real_obs['date'].min()
        numdays = len(real_obs)
        labels = list((datemin + datetime.timedelta(days=x)).strftime('%m-%d') for x in range(numdays))
        plt.xticks(list(range(numdays)), labels, rotation=60, fontsize=16)
        plt.yticks(fontsize=16)
        plt.xlabel('日期', fontsize=20)
        plt.ylabel('接触率', fontsize=20)
        plt.title(u'接触率与R0的变化', fontsize=25)
        plt.legend(['接触率', 'R0'], prop={'size': 20},
                    ncol=2, fancybox=True, shadow=True, loc='upper right')
        plt.show()

    def plot_fitted_result(self, real_obs: pandas.core.frame.DataFrame):
        fig, ax = plt.subplots(figsize=(17, 6))
        plt.plot(self.estimation['Time'], self.estimation['Estimated_Infected'], color='red',linewidth=3)
        plt.plot(self.estimation['Time'], real_obs['I'], color='orange',linewidth=3)
        plt.plot(self.estimation['Time'], self.estimation['Estimated_Exposed'], color='blue',linewidth=3)
        plt.plot(self.estimation['Time'], real_obs['E'], color='green',linewidth=3)
        # set x tricks
        datemin = real_obs['date'].min()
        numdays = len(real_obs)
        labels = list((datemin + datetime.timedelta(days=x)).strftime('%m-%d') for x in range(numdays))
        plt.xticks(list(range(numdays)), labels, rotation=60, fontsize=16)
        plt.yticks(fontsize=16)
        plt.xlabel('日期', fontsize=20)
        plt.ylabel('人数', fontsize=20)
        plt.title('使用SEIR拟合的值', fontsize=25)
        plt.legend(['估计感染人数', '实际感染人数', '估计潜伏人数', '实际潜伏人数'], prop={'size': 20},
                   ncol=4, fancybox=True, shadow=True, loc='upper left')
        plt.show()



## 模型评估
#############################################################################
def plot_test_data_with_MAPE_Infected(test, predict_data, title):
    """
    比较感染人数的观测值与预测值间的差距
    """
    y = test["I"].reset_index(drop=True)
    y_pred = predict_data[:len(test)]['Infected'].reset_index(drop=True)
    mape = np.mean(np.abs((y - y_pred)) / np.abs(y))
    print("The MAPE between obvervation and prediction of Infected is: ".format(mape))
    print("确诊人数观测值和预测值之间的MAPE为: ".format(mape))
    print(mape)

    # 画图
    fig, ax = plt.subplots(figsize=(17, 6))
    plt.plot(test['date'], y, linewidth=3,color='steelblue')
    plt.plot(test['date'], y_pred, linewidth=3,color='orangered')

    plt.tick_params(labelsize=16)
    plt.xlabel('日期', fontsize=22)
    plt.ylabel('感染病例', fontsize=20)
    plt.title(title, fontsize=25)
    plt.legend(['观测', '预测'], loc='upper right', prop={'size': 20},

               ncol=2, fancybox=True, shadow=True)

    plt.show()
#############################################################################
def plot_test_data_with_MAPE_Resistant(test, predict_data, title):
    # 比较移除人数的观测值与预测值间的差距

    y = test["R"].reset_index(drop=True)
    y_pred = predict_data[:len(test)]['Resistant'].reset_index(drop=True)
    mape = np.mean(np.abs((y - y_pred)) / np.abs(y))
    print("移除人数观测值和预测值之间的MAPE为: ".format(mape))
    print(mape)

    # 画图
    fig, ax = plt.subplots(figsize=(17, 6))
    plt.plot(test['date'], y, linewidth=3,color='coral')
    plt.plot(test['date'], y_pred, linewidth=3,color='blueviolet')

    plt.tick_params(labelsize=16)
    plt.xlabel('日期', fontsize=20)
    plt.ylabel('移除人数', fontsize=20)
    plt.title(title, fontsize=25)

    plt.legend(['观测', '预测'], loc='upper left', prop={'size': 20},
                ncol=2, fancybox=True, shadow=True)


    plt.show()

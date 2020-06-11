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
# �޸ı���Ϊ����
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']

class dynamic_SEIR:
    """
    'eons' (number of time points to model, default 1000) ��ģʱ��������

    'Susceptible' (number of susceptible individuals at time 0, default 950) 0ʱ���׸�����
    'Exposed' (number of individuals during incubation period) Ǳ��������
    'Infected' (number of infected individuals at time 0, default 50) 0ʱ�̸�Ⱦ����
    'Resistant' (number of resistant individuals at time 0, default 0) 0ʱ������/��������

    'rateSE'  S��E��ת���ʣ�Ĭ��0.05
    'rateEI'  E��I��ת���ʣ�Ĭ��0.1
    'rateIR'  I��Rת���ʣ�Ĭ��0.01
    ������rateSE��rateEI��rateIR�����ı仯����
    """

    def __init__(self, eons=1000, Susceptible=950, Exposed=100, Infected=50, Resistant=0, rateIR=0.01, rateEI=0.1,
                 alpha=0.3, c=5, b=-10, past_days=30):
        self.eons = eons  # Ԥ������
        self.Susceptible = Susceptible
        self.Exposed = Exposed
        self.Infected = Infected
        self.Resistant = Resistant
        self.rateSE = None
        self.rateIR = rateIR
        self.rateEI = rateEI
        self.numIndividuals = Susceptible + Infected + Resistant + Exposed  # S��E��I��R��������
        self.alpha = alpha
        self.c = c
        self.b = b
        self.past_days = past_days  # ���ϴι۲���������Ԥ��
        self.results = None
        self.modelRun = False

    def _calculate_beta(self, c: float, t: int, alpha: float, b: float, past_days: int):
        """
        ����ĳЩ��������beta
        """
        t = t + past_days
        return c * exp(-alpha * (t + b)) * pow((1 + exp(-alpha * (t + b))), -2)

    def run(self, death_rate):
        Susceptible = [self.Susceptible]
        Exposed = [self.Exposed]
        Infected = [self.Infected]
        Resistant = [self.Resistant]

        for i in range(1, self.eons):  # Ԥ�������
            self.rateSE = self._calculate_beta(c=self.c, t=i, b=self.b,
                                               alpha=self.alpha, past_days=self.past_days)

            S_to_E = (self.rateSE * Susceptible[-1] * Infected[-1]) / self.numIndividuals
            E_to_I = (self.rateEI * Exposed[-1])
            I_to_R = (Infected[-1] * self.rateIR)

            Susceptible.append(Susceptible[-1] - S_to_E)
            Exposed.append(Exposed[-1] + S_to_E - E_to_I)
            Infected.append(Infected[-1] + E_to_I - I_to_R)
            Resistant.append(Resistant[-1] + I_to_R)


        # ������ = ������ * �ָ�����(recovery group)
        Death = list(map(lambda x: (x * death_rate), Resistant))

        # ������ = �Ƴ���(removed) - ������
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

        plt.legend(['�׸�����', '��Ⱦ����', 'Ǳ������', '�Ƴ�����', '��������', '��������'], prop={'size': 12},
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
        # ���� x trick
        datemin = starting_point
        numdays = len(self.results)
        labels = list((datemin + datetime.timedelta(days=x)).strftime('%m-%d') for x in range(numdays))
        plt.xticks(list(range(numdays)), labels, rotation=60, fontsize=18)
        plt.yticks(fontsize=18)
        plt.xlabel(xlabel, fontsize=22)
        plt.ylabel(ylabel, fontsize=20)
        # plt.legend(['Infected', 'Removed', 'Exposed', 'Heal', 'Death'], prop={'size': 12},
        #            ncol=5, fancybox=True, shadow=True, loc='upper left')

        plt.legend(['��Ⱦ', '�Ƴ�', 'Ǳ��', '����', '����'], prop={'size': 14},
                              ncol=5, fancybox=True, shadow=True, loc='upper left')

        plt.title(title, fontsize=25)
        plt.show()



class Train_Dynamic_SEIR:
    """
    'eons' (number of time points to model, default 1000)
    'Susceptible' (number of susceptible individuals at time 0, default 950)  0ʱ���׸�Ⱦ������Ĭ��Ϊ950
    'Exposed' (number of individuals during incubation period)          0ʱ��Ǳ���ڵ�����
    'Infected' (number of infected individuals at time 0, default 50)   0ʱ���Ѹ�Ⱦ��������Ĭ��Ϊ50
    'Resistant' (number of resistant individuals at time 0, default 0)  0ʱ������/������������Ĭ��Ϊ0
    'rateSE' (base rate 'beta' from S to E, default 0.05)   ��S��E��betaֵ
    'rateIR' (base rate 'gamma' from I to R, default 0.01)  ��I��R��gammaֵ
    'rateEI' (base rate of isolation 'altha', from E to I, default 0.1)  ��E��I��alphaֵ
    """

    def __init__(self, data: pandas.core.frame.DataFrame,
                 population: int, epoch=1000, rateIR=0.01, rateEI=0.1, c=1, b=-3, alpha=0.1):
        self.epoch = epoch  # change weights in each epoch
        self.steps = len(data)
        # ʵ�ʹ۲�
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

        # SEIRģ������������
        self.c = c  # �����²�
        self.b = b  # �����²�
        self.alpha = alpha  # �����²�
        self.rateSE = self._calculate_beta(c=self.c, t=0, b=self.b, alpha=self.alpha)  # �����²�
        self.rateIR = rateIR
        self.rateEI = rateEI
        self.numIndividuals = population  # ������
        self.results = None
        self.estimation = None
        self.modelRun = False
        self.loss = None
        self.betalist = []

    """
    ����betaֵ
    """
    def _calculate_beta(self, c: float, t: int, alpha: float, b: float):

        return c * exp(-alpha * (t + b)) * pow((1 + exp(-alpha * (t + b))), -2)

    def _calculate_loss(self):
        """
        loss = ����lossƽ��֮�͵Ŀ���
        """
        return mean_squared_error(self.Infected, self.I_pre)

    def _calculate_MAPE(self):
        """
        ����ģ��(estimated)ֵ�����(fitted)ֵ֮���MAPE
        """
        y = np.array(self.Infected)
        y_pred = np.array(self.I_pre)
        mape = np.abs((y - y_pred)) / np.abs(y)
        return np.mean(mape)

    def _update(self):
        """
        ѵ������train()�İ�����������hepler function)
        ��һ�ε�����ʹ���ݶ��½������ҳ�ȫ����С����
        �������ݶȲ����²���
        """
        E = 2.71828182846
        alpha_eta = 0.000000000000001  # ѧϰ��
        b_eta = 0.00000000001  # ѧϰ��
        c_eta = 0.0000000000001  # ѧϰ��
        alpha_temp = 0.0
        c_temp = 0.0
        b_temp = 0.0
        for t in range(0, self.steps):
            formula = E ** (self.alpha * (t + self.b))
            formula2 = E ** (-self.alpha * (t + self.b))

            loss_to_beta = -2 * (self.Infected[t] - self.I_pre[t]) * (self.I_pre[t]) * t * self.Susceptible[
                t] / self.numIndividuals

            # ʹ�����������ƫ����
            beta_to_alpha = -self.c * formula * (t + self.b) * (formula - 1) * pow((1 + formula), -3)
            beta_to_b = -self.c * formula * self.alpha * (formula - 1) * pow((1 + formula), -3)
            beta_to_c = formula2 * pow((1 + formula2), -2)

            alpha_temp += loss_to_beta * beta_to_alpha  # ���ݶ�
            b_temp += loss_to_beta * beta_to_b  # ���ݶ�
            c_temp += loss_to_beta * beta_to_c  # ���ݶ�

        self.alpha -= alpha_eta * alpha_temp  # ����alpha��b��c
        self.b -= b_eta * b_temp
        self.c -= c_eta * c_temp
        # print('alpha��b��c��ֵ����: ')
        # print("c: {}, b: {}, alpha: {}".format(self.c, self.b, self.alpha))

    def train(self):
        """
        ʹ��ʵʱ���ݽ���SEIRģ�ͽ��й���
        ͨ��epoch�ĵ������ƹ��Ʋ���
        Ŀ��:
            ʹ���ݶ��½�ͨ����С��ʧ�����ҵ����beta���Ӵ��ʣ�

        �ݶ��½�:
                Ϊ�˽���ݶȣ�����ʹ���µ�alpha��c��bֵ�������ݵ㣬����ƫ����
                �µ��ݶȸ������ǳɱ�������cost function���ڵ�ǰλ�ã���ǰ����ֵ����б�ʺ�����Ӧ�ø��²����ķ���
                ���Ǹ��µĴ�С��ѧϰ�ʿ��ơ� ����μ������_update����������
        """
        for e in range(self.epoch):
            # Ԥ���б�
            self.S_pre = []
            self.E_pre = []
            self.I_pre = []
            self.R_pre = []

            self.rate_S_to_E = []
            self.rate_E_to_I = []
            self.rate_I_to_R = []


            # �𲽽���Ԥ��
            for t in range(0, self.steps):
                if t == 0:
                    self.S_pre.append(self.Susceptible[0])
                    self.E_pre.append(self.Exposed[0])
                    self.I_pre.append(self.Infected[0])
                    self.R_pre.append(self.Resistant[0])
                    self.rateSE = self._calculate_beta(c=self.c, t=t, b=self.b,
                                                       alpha=self.alpha)
                    # print("time {}, beta {}".format(t, self.rateSE))

                    # �洢�����ϵ�Beta
                    if e == (self.epoch - 1):
                        self.betalist.append(self.rateSE)

                else:
                    self.rateSE = self._calculate_beta(c=self.c, t=t, b=self.b,
                                                       alpha=self.alpha)
                    # print("time {}, beta {}".format(t, self.rateSE))

                    # �ռ������ϵ�Beta
                    if e == (self.epoch - 1):
                        self.betalist.append(self.rateSE)

                    # ��ʵʱ����Ӧ����SEIRģ��
                    S_to_E = (self.rateSE * self.Susceptible[t] * self.Infected[t]) / self.numIndividuals
                    E_to_I = (self.rateEI * self.Exposed[t])
                    I_to_R = (self.Infected[t] * self.rateIR)
                    self.S_pre.append(self.Susceptible[t] - S_to_E)
                    self.E_pre.append(self.Exposed[t] + S_to_E - E_to_I)
                    self.I_pre.append(self.Infected[t] + E_to_I - I_to_R)
                    self.R_pre.append(self.Resistant[t] + I_to_R)


            # ��¼���һ�ε���ʱ�Ĺ���ֵ
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

                print('ȫ��MAPEֵΪ{}'.format(MAPE))
                print('��ѽӴ���Ϊ{}'.format(self.rateSE))

            # ����ÿ�ε����е���ʧ
            self.loss = self._calculate_loss()
            # print("The loss in iteration {} is {}".format(e, self.loss))
            # print("Current beta is {}".format(self.rateSE))

            # �����ѧϰ���Ż�����
            self._update()  # ��ÿ��������ʹ�á��ݶ��½������²���

        return self.estimation  # ���µĹ���ֵ

    def plot_fitted_beta_R0(self, real_obs: pandas.core.frame.DataFrame):
        fig, ax = plt.subplots(figsize=(17, 6))
        plt.plot(self.estimation['Time'],  self.betalist, linewidth=3,color='firebrick')# ���ƽӴ���
        Rlist = [x / self.rateIR for x in self.betalist]  # transmissibility over time
        plt.plot(self.estimation['Time'], Rlist, linewidth=3,color='steelblue')

        # set x tricks
        datemin = real_obs['date'].min()
        numdays = len(real_obs)
        labels = list((datemin + datetime.timedelta(days=x)).strftime('%m-%d') for x in range(numdays))
        plt.xticks(list(range(numdays)), labels, rotation=60, fontsize=16)
        plt.yticks(fontsize=16)
        plt.xlabel('����', fontsize=20)
        plt.ylabel('�Ӵ���', fontsize=20)
        plt.title(u'�Ӵ�����R0�ı仯', fontsize=25)
        plt.legend(['�Ӵ���', 'R0'], prop={'size': 20},
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
        plt.xlabel('����', fontsize=20)
        plt.ylabel('����', fontsize=20)
        plt.title('ʹ��SEIR��ϵ�ֵ', fontsize=25)
        plt.legend(['���Ƹ�Ⱦ����', 'ʵ�ʸ�Ⱦ����', '����Ǳ������', 'ʵ��Ǳ������'], prop={'size': 20},
                   ncol=4, fancybox=True, shadow=True, loc='upper left')
        plt.show()



## ģ������
#############################################################################
def plot_test_data_with_MAPE_Infected(test, predict_data, title):
    """
    �Ƚϸ�Ⱦ�����Ĺ۲�ֵ��Ԥ��ֵ��Ĳ��
    """
    y = test["I"].reset_index(drop=True)
    y_pred = predict_data[:len(test)]['Infected'].reset_index(drop=True)
    mape = np.mean(np.abs((y - y_pred)) / np.abs(y))
    print("The MAPE between obvervation and prediction of Infected is: ".format(mape))
    print("ȷ�������۲�ֵ��Ԥ��ֵ֮���MAPEΪ: ".format(mape))
    print(mape)

    # ��ͼ
    fig, ax = plt.subplots(figsize=(17, 6))
    plt.plot(test['date'], y, linewidth=3,color='steelblue')
    plt.plot(test['date'], y_pred, linewidth=3,color='orangered')

    plt.tick_params(labelsize=16)
    plt.xlabel('����', fontsize=22)
    plt.ylabel('��Ⱦ����', fontsize=20)
    plt.title(title, fontsize=25)
    plt.legend(['�۲�', 'Ԥ��'], loc='upper right', prop={'size': 20},

               ncol=2, fancybox=True, shadow=True)

    plt.show()
#############################################################################
def plot_test_data_with_MAPE_Resistant(test, predict_data, title):
    # �Ƚ��Ƴ������Ĺ۲�ֵ��Ԥ��ֵ��Ĳ��

    y = test["R"].reset_index(drop=True)
    y_pred = predict_data[:len(test)]['Resistant'].reset_index(drop=True)
    mape = np.mean(np.abs((y - y_pred)) / np.abs(y))
    print("�Ƴ������۲�ֵ��Ԥ��ֵ֮���MAPEΪ: ".format(mape))
    print(mape)

    # ��ͼ
    fig, ax = plt.subplots(figsize=(17, 6))
    plt.plot(test['date'], y, linewidth=3,color='coral')
    plt.plot(test['date'], y_pred, linewidth=3,color='blueviolet')

    plt.tick_params(labelsize=16)
    plt.xlabel('����', fontsize=20)
    plt.ylabel('�Ƴ�����', fontsize=20)
    plt.title(title, fontsize=25)

    plt.legend(['�۲�', 'Ԥ��'], loc='upper left', prop={'size': 20},
                ncol=2, fancybox=True, shadow=True)


    plt.show()

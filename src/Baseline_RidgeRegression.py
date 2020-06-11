# -*-coding: GBK -*-

import numpy as np
import operator
import matplotlib.pyplot as plt
import pandas as pd

import datetime
import seaborn as sns
import matplotlib.dates as mdates
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
# Import models from scikit learn module
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV, LinearRegression, ElasticNetCV
from src.helper_fun_model import *
import warnings
from mpl_toolkits.basemap import Basemap
from matplotlib.pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False # ����������ʾ����
warnings.filterwarnings('ignore')


# ��ȡ����

df = pd.read_csv('data/DXYArea.csv')
# ������ϴ
df['date'] = pd.to_datetime(df['date'])
df = df[df['date'] != df['date'].max()]

# ѡȡ2020��1��23��֮�������
df = df[df['date'] >= '2020-01-23']

china_df = get_China_total(df)

features_to_engineer = ['confirmed']

china_df = feature_engineering(china_df, features_to_engineer)
china_df.tail(2)


# _________________________�������5 �ĳ��� 7
# ���ﴫ�����ֵ����Ԥ��ӳ���ִ�����𣬽�������n������
Train, Test = split_train_test_by_date(china_df, 14)

regressors = ['Days', 'suspected'] # ģ������һ������

X_train = Train.loc[:, regressors + [x+'_lag1' for x in features_to_engineer]]
y_train = Train['confirmed']
X_test = Test.loc[:, regressors + [x+'_lag1' for x in features_to_engineer]]
y_test = Test['confirmed']

# ģ�ͻع�
# print��X_train���б�ǩ
X_train.columns

# ʹ��StandardScaler���б任
sc = StandardScaler().fit(X_train)
X_train_sc = sc.transform(X_train)
X_test_sc = sc.transform(X_test)

X_train_sc[:3]

# ���Իع�

linear = LinearRegression()
linear.fit(X_train_sc, y_train)
# validation score
get_validation_score(linear, X_test_sc, y_test)

# LASSO regression

# fine from default hyperparameters��Ĭ�ϳ������п���
lasso = LassoCV(cv=10)
lasso.fit(X_train_sc, y_train)
alpha = lasso.alpha_
print("basic alpha...", alpha)

# find from more refined list
print("Try again for more precision with alphas centered around " + str(alpha))
lasso = LassoCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8,
                          alpha * .85, alpha * .9, alpha * .95, alpha, alpha * 1.05,
                          alpha * 1.1, alpha * 1.15, alpha * 1.25, alpha * 1.3, alpha * 1.35,
                          alpha * 1.4],
                max_iter = 100000, cv = 5) #--------------�Ӵ�����������۲�ģ��Ч��------------------
lasso.fit(X_train_sc, y_train)
alpha = lasso.alpha_
print("Best alpha :", alpha)

# validation score
get_validation_score(lasso, X_test_sc, y_test)


print("---------------------")


# Ridge Regression
ridge = RidgeCV(cv=5)
ridge.fit(X_train_sc, y_train)
alpha = ridge.alpha_
print("basic alpha: ", alpha)
print("Try again for more precision with alphas centered around " + str(alpha))
ridge = RidgeCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8,
                          alpha * .85, alpha * .9, alpha * .95, alpha, alpha * 1.05,
                          alpha * 1.1, alpha * 1.15, alpha * 1.25, alpha * 1.3, alpha * 1.35,
                          alpha * 1.4], cv = 5)
ridge.fit(X_train_sc, y_train)
alpha = ridge.alpha_
print("Best alpha :", alpha)
# validation score
get_validation_score(lasso, X_test_sc, y_test)


print("---------------------")

# ------------------------��������CSV�ļ�------------------------

train_index = X_train.index
test_index = X_test.index

# ����dataframe
X_train = pd.DataFrame(X_train_sc, columns=X_train.columns, index=train_index)
X_test = pd.DataFrame(X_test_sc, columns=X_train.columns, index=test_index)

# ����Ԥ��ֵ
predictions_ridge = np.squeeze(ridge.predict(X_test))
# ����в�
residuals_ridge = np.squeeze(y_test) - predictions_ridge

# ����dataframe
d = {'y_test': np.squeeze(y_test), 'predictions': predictions_ridge, 'residuals_ridge': residuals_ridge}
df_residuals_ridge = pd.DataFrame(data=d, index=test_index)
df_residuals_ridge['date'] = Test['date']
# df_residuals_ridge.shape
# �ѽ��д��csv�ļ�
df_residuals_ridge.to_csv('results/residuals_ridge_test.csv', index_label='test_index')



# ����Ԥ��ֵ
predictions_ridge = np.squeeze(ridge.predict(X_train))
# ����в�
residuals_ridge = np.squeeze(y_train) - predictions_ridge
# ����dataframe
d = {'y_train': np.squeeze(y_train), 'predictions': predictions_ridge, 'residuals_ridge': residuals_ridge}
df_residuals_ridge = pd.DataFrame(data=d, index=train_index)
df_residuals_ridge['date'] = Train['date']
# df_residuals_ridge.shape
# д��csv�ļ�
df_residuals_ridge.to_csv('results/residuals_ridge_train.csv', index_label='train_index')


df_residuals_ridge.head(2)


# ��ع�ķ���

# ���ϻ�ͼ
sns.jointplot('y_train', 'residuals_ridge', data=df_residuals_ridge, kind="reg")
plt.tick_params(labelsize=16)
plt.xlabel('y_train', fontsize=20)
plt.ylabel('residuals_ridge', fontsize=20)

plt.plot()
plt.show()



# ����ع����Ԥ��
ridge_res_train = pd.read_csv('results/residuals_ridge_train.csv')
del ridge_res_train['train_index']
ridge_res_test = pd.read_csv('results/residuals_ridge_test.csv')
del ridge_res_test['test_index']

ridge_res_test.Timestamp = pd.to_datetime(ridge_res_test.date,format='%Y-%m-%d %H:%M')
ridge_res_test.index = ridge_res_test.Timestamp
ridge_res_test = ridge_res_test.sort_index()

ridge_res_train.Timestamp = pd.to_datetime(ridge_res_train.date,format='%Y-%m-%d %H:%M')
ridge_res_train.index = ridge_res_train.Timestamp
ridge_res_train = ridge_res_train.sort_index()

ridge_all = pd.concat([ridge_res_train,ridge_res_test])
ridge_all.head(3)

# ����������MAPE
y = ridge_res_test['y_test'][:7]
y_pred = ridge_res_test['predictions'][:7]
mape = np.abs((y - y_pred)) / np.abs(y)
print(np.mean(mape))


# ��ͼ
ridge_res_train['predictions'].plot(figsize=(17, 6),
                                    fontsize=25,
                                    # label= '���ֵ',
                                    linewidth=2,
                                    color = 'red')
ridge_res_train['y_train'].plot(figsize=(17, 6),
                                # label= '�۲�ֵ',
                                linewidth=2, color = 'steelblue')
plt.title('��ع鷨Ԥ���ȫ����ȷ�ﲡ��������1��22��-3��1�գ�', fontsize=25)
plt.tick_params(labelsize=16)
plt.ylabel('ȷ�ﲡ����', fontsize=20)
plt.xlabel('����', fontsize=20)
plt.legend(['���', '�۲�'], loc='upper left', prop={'size': 16}, ncol=2, fancybox=True, shadow=True)
plt.show()



ridge_res_test['y_test'].plot(figsize=(17,6),
                            linewidth=2,
                            color = 'green')

ridge_res_test['predictions'].plot(figsize=(17,6),
                                   color = 'purple',
                                   linewidth=2)

plt.title('��ع鷨Ԥ���ȫ����ȷ�ﲡ������', fontsize=25)

plt.tick_params(labelsize=18)
plt.ylabel('ȷ�ﲡ����', fontsize=20)
plt.xlabel('����', fontsize=20)
plt.legend(['�۲�', '��ع�Ԥ��'], loc='upper left', prop={'size': 16}, ncol=2, fancybox=True, shadow=True)
plt.show()



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
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号
warnings.filterwarnings('ignore')


# 读取数据

df = pd.read_csv('data/DXYArea.csv')
# 数据清洗
df['date'] = pd.to_datetime(df['date'])
df = df[df['date'] != df['date'].max()]

# 选取2020年1月23日之后的数据
df = df[df['date'] >= '2020-01-23']

china_df = get_China_total(df)

features_to_engineer = ['confirmed']

china_df = feature_engineering(china_df, features_to_engineer)
china_df.tail(2)


# _________________________将这里的5 改成了 7
# 这里传入的数值代表预测从程序执行日起，接下来的n天走势
Train, Test = split_train_test_by_date(china_df, 14)

regressors = ['Days', 'suspected'] # 模型中另一个变量

X_train = Train.loc[:, regressors + [x+'_lag1' for x in features_to_engineer]]
y_train = Train['confirmed']
X_test = Test.loc[:, regressors + [x+'_lag1' for x in features_to_engineer]]
y_test = Test['confirmed']

# 模型回归
# print出X_train的列标签
X_train.columns

# 使用StandardScaler进行变换
sc = StandardScaler().fit(X_train)
X_train_sc = sc.transform(X_train)
X_test_sc = sc.transform(X_test)

X_train_sc[:3]

# 线性回归

linear = LinearRegression()
linear.fit(X_train_sc, y_train)
# validation score
get_validation_score(linear, X_test_sc, y_test)

# LASSO regression

# fine from default hyperparameters从默认超参数中可以
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
                max_iter = 100000, cv = 5) #--------------加大迭代次数，观察模型效果------------------
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

# ------------------------保存结果到CSV文件------------------------

train_index = X_train.index
test_index = X_test.index

# 创建dataframe
X_train = pd.DataFrame(X_train_sc, columns=X_train.columns, index=train_index)
X_test = pd.DataFrame(X_test_sc, columns=X_train.columns, index=test_index)

# 计算预测值
predictions_ridge = np.squeeze(ridge.predict(X_test))
# 计算残差
residuals_ridge = np.squeeze(y_test) - predictions_ridge

# 创建dataframe
d = {'y_test': np.squeeze(y_test), 'predictions': predictions_ridge, 'residuals_ridge': residuals_ridge}
df_residuals_ridge = pd.DataFrame(data=d, index=test_index)
df_residuals_ridge['date'] = Test['date']
# df_residuals_ridge.shape
# 把结果写入csv文件
df_residuals_ridge.to_csv('results/residuals_ridge_test.csv', index_label='test_index')



# 计算预测值
predictions_ridge = np.squeeze(ridge.predict(X_train))
# 计算残差
residuals_ridge = np.squeeze(y_train) - predictions_ridge
# 创建dataframe
d = {'y_train': np.squeeze(y_train), 'predictions': predictions_ridge, 'residuals_ridge': residuals_ridge}
df_residuals_ridge = pd.DataFrame(data=d, index=train_index)
df_residuals_ridge['date'] = Train['date']
# df_residuals_ridge.shape
# 写入csv文件
df_residuals_ridge.to_csv('results/residuals_ridge_train.csv', index_label='train_index')


df_residuals_ridge.head(2)


# 岭回归的分析

# 联合画图
sns.jointplot('y_train', 'residuals_ridge', data=df_residuals_ridge, kind="reg")
plt.tick_params(labelsize=16)
plt.xlabel('y_train', fontsize=20)
plt.ylabel('residuals_ridge', fontsize=20)

plt.plot()
plt.show()



# 用岭回归进行预测
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

# 测试用例的MAPE
y = ridge_res_test['y_test'][:7]
y_pred = ridge_res_test['predictions'][:7]
mape = np.abs((y - y_pred)) / np.abs(y)
print(np.mean(mape))


# 画图
ridge_res_train['predictions'].plot(figsize=(17, 6),
                                    fontsize=25,
                                    # label= '拟合值',
                                    linewidth=2,
                                    color = 'red')
ridge_res_train['y_train'].plot(figsize=(17, 6),
                                # label= '观测值',
                                linewidth=2, color = 'steelblue')
plt.title('岭回归法预测的全国净确诊病例总数（1月22日-3月1日）', fontsize=25)
plt.tick_params(labelsize=16)
plt.ylabel('确诊病例数', fontsize=20)
plt.xlabel('日期', fontsize=20)
plt.legend(['拟合', '观测'], loc='upper left', prop={'size': 16}, ncol=2, fancybox=True, shadow=True)
plt.show()



ridge_res_test['y_test'].plot(figsize=(17,6),
                            linewidth=2,
                            color = 'green')

ridge_res_test['predictions'].plot(figsize=(17,6),
                                   color = 'purple',
                                   linewidth=2)

plt.title('岭回归法预测的全国净确诊病例总数', fontsize=25)

plt.tick_params(labelsize=18)
plt.ylabel('确诊病例数', fontsize=20)
plt.xlabel('日期', fontsize=20)
plt.legend(['观测', '岭回归预测'], loc='upper left', prop={'size': 16}, ncol=2, fancybox=True, shadow=True)
plt.show()



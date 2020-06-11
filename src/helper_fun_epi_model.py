import numpy as np
import scipy.optimize as optimization
import pandas as pd
import pandas


#############################
## 数据处理
##############################

def get_province_df(df, provinceName: str) -> pandas.core.frame.DataFrame:
    """
    给定省的返回时间序列数据
    """
    return df[(df['province'] == provinceName) & (df['city'].isnull())]


def get_China_total(df) -> pandas.core.frame.DataFrame:
    """
    返回中国（包括香港和台湾）总计的时间序列数据
    """
    return df[(df['countryCode'] == 'CN') & (df['province'].isnull())]


def get_China_exclude_province(df, provinceName: str) -> pandas.core.frame.DataFrame:
    """
    返回中国总计的时间序列数据(不包括给定的省)
    """
    Hubei = get_province_df(df, provinceName)
    China_total = get_China_total(df)

    NotHubei = China_total.reset_index(drop=True)
    Hubei = Hubei.reset_index(drop=True)

    NotHubei['E'] = NotHubei['E'] - Hubei['E']
    NotHubei['R'] = NotHubei['R'] - Hubei['R']
    NotHubei['I'] = NotHubei['I'] - Hubei['I']

    return NotHubei
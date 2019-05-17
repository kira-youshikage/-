'''
两种数据标准化方法，min_max_scale是0-1标准化，使得数据在0-1区间内
z_core标准化使得数据的均值为0，方差为1
'''
"""
第一次修改内容：
    代码规范化
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

def min_max_scale(data_mat):
	scaler = MinMaxScaler()
	scaler.fit(data_mat)
	scaler.data_max_
	data_scale_mat = scaler.transform(data_mat)
	a = scaler.inverse_transform(data_scale_mat)
	return data_scale_mat

def z_core(data_mat):
	scaler = StandardScaler()
	data_scale_mat = scaler.fit_transform(data_mat)
	a = scaler.inverse_transform(data_scale_mat)
	return data_scale_mat



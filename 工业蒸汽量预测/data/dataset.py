import numpy as np
import txt
import math
import normalization
from sklearn.decomposition import PCA, IncrementalPCA
import ANOVA
import LOF
from config import DefaultConfig

opt = DefaultConfig()

# 获得清洗后的数据
def get_data(train = True, test = False):
	if test == False:
		data = txt.read_data(opt.train_data_address)
		data.replace(' ','')
		data = data.split('\n')
		n = len(data)-1
		data_list = []
		for i in range(n):
			row_data = data[i].split('\t')
			data_list.append(row_data)
		data_np = np.float32(np.array(data_list[1:][:]))
		np.random.shuffle(data_np)
		n = math.floor(data_np.shape[0] * opt.train_n)
		if train == True:
			train_data = np.array(
				data_np[:n, :data_np.shape[1] - 1])
			train_label = np.array(
				data_np[:n, data_np.shape[1] - 1])
			train_label = train_label.reshape([train_data.shape[0], 1])
			return train_data, train_label
		if train == False:
			validation_data = np.array(
				data_np[n:, :data_np.shape[1] - 1])
			validation_label = np.array(
				data_np[n:, data_np.shape[1] - 1])
			validation_label = validation_label.reshape([validation_data.shape[0], 1])
			return validation_data, validation_label
	else:
		data = txt.read_data(opt.test_data_address)
		data.replace(' ','')
		data = data.split('\n')
		n = len(data)-1
		data_list = []
		for i in range(n):
			row_data = data[i].split('\t')
			data_list.append(row_data)
		test_data = np.float32(np.array(data_list[1:][:]))
		test_label = np.float32(np.arange(test_data.shape[0]))
		test_label = test_label.reshape([test_data.shape[0], 1])
		return test_data, test_label

# 标准化
def scale(train_data,validation_data, test_data, type = opt.scale_type):
	train_n = train_data.shape[0]
	validation_n = validation_data.shape[0]
	data_np = np.row_stack((train_data, validation_data, test_data))
	if type == 'min_max':
		transform = normalization.min_max_scale
	elif type == 'z_core':
		transform = normalization.z_core
	else:
		print("标准化的类型输入错误")
		return
	data_scale = transform(data_np)
	train_scale_data = np.float32(data_scale[:train_n, :])
	validation_scale_data = np.float32(
		data_scale[train_n:train_n + validation_n, :])
	test_scale_data = np.float32(
		data_scale[train_n + validation_n:,:])
	return train_scale_data, validation_scale_data, test_scale_data

# 排除异常点
def remove(train_data, train_label, k = opt.k):
	train_remove = np.column_stack((train_data, train_label))
	LOF_list = LOF.LOF(train_remove, k)
	train_data = list(train_data)
	train_label = list(train_label)
	train_remove_data = []
	train_remove_label = []
	for i in range(len(LOF_list)):
		if LOF_list[i][1] <= opt.lof:
			train_remove_data.append(train_data[i][:])
			train_remove_label.append(train_label[i])
	train_remove_data = np.array(train_remove_data)
	train_remove_label = np.array(train_remove_label)
	train_remove_label = train_remove_label.reshape(
		[train_remove_data.shape[0], 1])
	return train_remove_data, train_remove_label

def pca(train_data, validation_data, test_data, pca_n = opt.pca_n, pca_1 = opt.pca):
	train_n = train_data.shape[0]
	validation_n = validation_data.shape[0]
	data = np.row_stack((train_data, validation_data, test_data))
	if pca_1 == 'PCA':
		pca_1 = PCA(n_components = pca_n,svd_solver='randomized')
	elif pca_1 == 'incrementalPCA':
		pca_1 = IncrementalPCA(n_components = pca_n)
	data = pca_1.fit_transform(data)
	train_PCA_data = np.array(data[:train_n,:])
	validation_PCA_data = np.array(data[train_n:train_n + validation_n,:])
	test_PCA_data = np.array(data[train_n + validation_n:,:])
	return train_PCA_data, validation_PCA_data, test_PCA_data


def anova(train_data, train_label, validation_data, validation_label, test_data):
	train_n = train_data.shape[0]
	train_nc = train_data.shape[1]
	data = np.row_stack((train_data, validation_data))
	label = np.row_stack((train_label, validation_label))
	data = np.column_stack((data, label))
	anova_result = ANOVA.ANOVA(data)
	head_list = []
	train_anova_data = []
	validation_anova_data = []
	test_anova_data = []
	for i in range(train_nc):
		index = 'A' + str(i)
		F = anova_result['F'][index]
		if F > opt.F:
			string = '第' + str(i + 1) + '特征'
			head_list.append(string)
			train_anova_data.append(list(train_data[:,i]))
			validation_anova_data.append(list(validation_data[:,i]))
			test_anova_data.append(list(test_data[:,i]))
	train_anova_data = np.array(train_anova_data).T
	validation_anova_data = np.array(validation_anova_data).T
	test_anova_data = np.array(test_anova_data).T
	return train_anova_data, validation_anova_data, test_anova_data
	#return head_list, train_anova_data, validation_anova_data, test_anova_data

def delete_character(train_data, validation_data, test_data, character = opt.character):
	train_n = train_data.shape[0]
	validation_n = validation_data.shape[0]
	data = np.row_stack((train_data, validation_data, test_data))
	data_list = []
	for i in range(data.shape[1]):
		if i not in character:
			data_list.append(data[:, i])
	data = np.reshape(np.array(data_list).T, [data.shape[0], data.shape[1] - len(character)])
	train_data = data[:train_n, :]
	validation_data = data[train_n:train_n + validation_n, :]
	test_data = data[train_n + validation_n:, :]
	return train_data, validation_data, test_data
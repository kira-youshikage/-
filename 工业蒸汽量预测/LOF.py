'''
利用局部异常因子（LOF）来找出异常点
'''
"""
第一次修改内容：
    代码规范化
"""
#准备工作
from collections import defaultdict
import numpy as np
import xlrd
import xlwt

import os 
import matplotlib.pyplot as plt
import heapq
from sklearn.metrics import pairwise_distances

#数据点
instances = np.matrix([[0,0],[0,1],[1,1],[1,0],[5,0]])


#计算K距离邻居:
def all_indices(value,inlist):
	out_indices = []
	idx = -1
	while True:
		try:
			idx = inlist.index(value,idx + 1)
			out_indices.append(idx)
		except ValueError:
			break
	return out_indices

def LOF(instances,k):
	#操作方法
	#获取点两两之间的距离pairwise_distances
	k = k
	distance = 'manhattan'
	dist = pairwise_distances(instances,metric=distance)
	#计算K距离，使用heapq来获得K最近邻
	#计算K距离
	k_distance = defaultdict(tuple)
	#对每个点进行计算
	for i in range(instances.shape[0]):
		#获得它和其他所有点之间的距离
		#为了方便，将数组转为列表
		distances = dist[i].tolist()
		#获得K最近邻
		ksmallest = heapq.nsmallest(k + 1,distances)[1:][k - 1]
		#获取它们的索引号
		ksmallest_idx = distances.index(ksmallest)
		#记录下每个点的第K给最近邻以及到它的距离
		k_distance[i] = (ksmallest,ksmallest_idx)
	#计算K距离邻居
	k_distance_neig = defaultdict(list)
	#对每个点进行计算
	for i in range(instances.shape[0]):
		#获取它到所有邻居点的距离
		distances = dist[i].tolist()
		#print("k distance neighbourhood",i)
		#print(distances)
		#获取从第1到第K的最近邻
		ksmallest = heapq.nsmallest(k + 1,distances)[1:]
		#print(ksmallest)
		ksmallest_set = set(ksmallest)
		#print(ksmallest_set)
		ksmallest_idx = []
		#获取K里最小的元素的索引号
		for x in ksmallest_set:
			ksmallest_idx.append(all_indices(x,distances))
		#将列表的列表转为列表
		ksmallest_idx = [item for sublist in ksmallest_idx for item in sublist]
		#对每个点保存其K距离邻居
		k_distance_neig[i].extend(zip(ksmallest,ksmallest_idx))
	#计算可达距离和LRD:
	#局部可达密度
	local_reach_density = defaultdict(float)
	for i in range(instances.shape[0]):
		#LRD的分子，K距离邻居的个数
		no_neighbours = len(k_distance_neig[i])
		denom_sum = 0
		#可达距离求和
		for neigh in k_distance_neig[i]:
			#P的K距离和P与Q的距离中的最大者
			denom_sum += max(k_distance[neigh[1]][0],neigh[0])
		local_reach_density[i] = no_neighbours / (1.0 * denom_sum)
	#计算LOF
	lof_list=[]
	#计算局部异常因子
	#越接近1说明p的其邻域点密度差不多，p可能和邻域同属一簇;越大于1，说明p的密度小于其邻域点密度，p越可能是异常点。 
	for i in range(instances.shape[0]):
		lrd_sum = 0
		rdist_sum = 0
		for neigh in k_distance_neig[i]:
			lrd_sum += local_reach_density[neigh[1]]
			rdist_sum += max(k_distance[neigh[1]][0],neigh[0])
		lof_list.append((i,lrd_sum*rdist_sum))
	return lof_list



'''
if __name__ == '__main__':
	#数据点
	data = get_excel_data()
	k = 2
	lof_list = LOF(data,k)
	#x,y = get_x_y(lof_list)
	#plt.plot(y,'ro')
	#plt.show()
	#判断删除哪些点
	counter = 0
	size_r,size_c = data.shape
	for i in range(size_r):
		if lof_list[i][1]>5:
			counter = counter+1
			data[i,0] = 999
	print(counter / size_r)
	for i in range(counter):
		for j in range(len(data)):
			if  (data[j,0] == '999'):
				data = np.delete(data,j,axis = 0)
				break
	print(size_r)
	print(counter)
	#将数据写入Excel
	size_r,size_c = data.shape
	workbook = xlwt.Workbook(encoding = 'ascii')
	worksheet = workbook.add_sheet('结果')
	for i in range(size_r):
		for j in range(size_c):
			worksheet.write(i+1, j, label = data[i,j])
	workbook.save('排除异常点结果.xls')
'''
 

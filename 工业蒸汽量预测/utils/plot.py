import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config import DefaultConfig

opt = DefaultConfig()

def distribution_view(train_data, validation_data, test_data):
	train = np.row_stack((train_data, validation_data))
	n = train_data.shape[1]
	plt.figure(1)
	plt_number = [5,8,0]
	for i in range(n):
		plt.subplot(plt_number[0], plt_number[1], i + 1)
		sns.distplot(train[:, i], hist = False, rug = False,label = 'train' + str(i))
		sns.distplot(test_data[:,i], hist = False, rug = False,label = 'test')
	plt.show()


class DefaultConfig(object):
	# 数据地址
	train_data_address = 'data/zhengqi_train.txt'
	test_data_address = 'data/zhengqi_test.txt'
	# 训练集规模
	train_n = 0.9999
	# 降维参数(累积贡献度)
	# 'incrementalPCA' 或 'PCA'
	pca = 'PCA'
	pca_n = 31
	# 标准化参数
	scale_type = 'min_max'
	#scale_type = 'z_core'
	# LOF参数
	k = 2
	lof = 30
	# 方差分析参数F
	F = 35
	# 删除特征
	character = [5, 6,17, 20, 22, 11, 27]

	# sklearn_mlp模型参数
	#sklearn_solver = "adam"
	#sklearn_solver = "sgd"
	sklearn_solver = "lbfgs"
	sklearn_alpha = 0.27
	sklearn_hidden_layer_sizes = (100)
	#sklearn_hidden_layer_sizes = (30,20,15,10,5)
	sklearn_random_state = 1
	# ('identity', 'logistic', 'tanh', 'relu')
	sklearn_activation = 'tanh'

	# torch_mlp模型参数
	#torch_p_1 = 0.5
	#torch_p_2 = 0.5
	#torch_p_3 = 0.5
	torch_p_1 = 0
	torch_p_2 = 0
	torch_p_3 = 0
	#torch_optimizer = 'Momentum'
	torch_moment = 0.5
	torch_optimizer = 'SGD'
	#torch_optimizer = 'Adam'
	torch_learning_rate = 0.01
	torch_batch_size = 200
	torch_epoch_time = 180
	torch_time = 10

	# 回归分析参数
	#多项式回归参数
	features = 1
	#线性模型标准化
	linear_normalize = True

	# 工具参数
	txt_path = 'utils/test_label.txt'

import data
import model
import utils
import numpy as np
from sklearn.metrics import mean_squared_error as mse

train_data, train_label = data.get_data(train = True)
validation_data, validation_label = data.get_data(train = False, test = False)
test_data, test_label = data.get_data(train = False, test = True)

# 查看分布
#utils.distribution_view(train_data, validation_data, test_data)

# 删除特征
train_data, validation_data, test_data = data.delete_character(
	train_data, validation_data, test_data)

# 排除异常点
#train_data, train_label = data.remove(train_data, train_label)

# 降维
train_data, validation_data, test_data = data.pca(train_data, validation_data, test_data)

# 方差分析
#train_data, validation_data, test_data = data.anova(train_data, train_label, 
#	validation_data, validation_label, test_data)

# 标准化
train_data, validation_data, test_data = data.scale(train_data, validation_data, test_data)
'''
mlp_model = model.sklearn_mlp()
mlp_model.train(train_data, train_label, validation_data, validation_label)
mlp_test_label = mlp_model.predict(test_data)
utils.write_txt(mlp_test_label)
predict = mlp_model.predict(validation_data)
'''


'''
torch_mlp = model.torch_train(train_data, train_label, validation_data, validation_label)
test_label = model.torch_predict(test_data, torch_mlp)
predict_torch = model.torch_predict(validation_data, torch_mlp)
for i in range(predict.shape[0]):
	print(str(predict[i]) + str(predict_torch[i]) + str(validation_label[i]))
'''



# 线性回归
linear_model = model.Linear_model_build(train_data, train_label)
predict_linear = linear_model.predict(validation_data)
print("linear_loss：" + str(mse(predict_linear, validation_label)))
linear_test_label = linear_model.predict(test_data)
utils.write_txt(linear_test_label)

predict = linear_model.predict(train_data)
delete_list = []
for i in range(train_data.shape[0]):
	if abs(predict[i] - train_label[i]) > 1:
		delete_list.append(i)
data = []
label = []
for i in range(train_data.shape[0]):
	if i not in delete_list:
		data.append(list(train_data[i]))
		label.append(train_label[i])
train_data = np.array(data)
train_label = np.reshape(np.array(label),[-1,1])
linear_model = model.Linear_model_build(train_data, train_label)
predict_linear = linear_model.predict(validation_data)
print("linear_loss：" + str(mse(predict_linear, validation_label)))
linear_test_label = linear_model.predict(test_data)
utils.write_txt(linear_test_label)



'''
test_label = []
for i in range(linear_test_label.shape[0]):
	if linear_test_label[i, 0] > -3:
		test_label.append(linear_test_label[i, 0])
	else:
		test_label.append(mlp_test_label[i, 0])
test_label = np.reshape(np.array(test_label), [linear_test_label.shape[0],1])
utils.write_txt(test_label)
'''

# 多项式回归
'''
poly_model, train_data, validation_data, test_data = model.polynomial_model_build(
	train_data, validation_data, test_data, train_label, 1)
predict_poly = poly_model.predict(validation_data)
print(mse(predict_poly, validation_label))
'''
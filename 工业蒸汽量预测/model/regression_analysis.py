#回归分析
"""
第一次修改内容：
    代码规范化
"""
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import math
from config import DefaultConfig

opt = DefaultConfig()

#模型评估指标(均方差)
def MSE(test_target, predict_target):
	mes = mean_squared_error(test_target, predict_target)
	print("mse = %0.2f"%(mes))
	return mes

#模型评估指标(残差图)
def plot_residual(real_y,predicted_y):
	plt.cla()
	plt.xlabel("Predicted Y")
	plt.ylabel("Residual")
	plt.title("Residual Plot")
	plt.figure(1)
	diff = real_y - predicted_y
	plt.plot(predicted_y,diff,'go')
	plt.show()

#查看模型系数
def model_coef_view(model):
	model_coef = model.coef_
	print(model_coef)
	return model_coef

#模型测试函数
def model_test(test_x, test_y,model):
	predict_y = model.predict(test_x)
	#输出均方差
	mse=MSE(test_y, predict_y)
	#作残差图
	plot_residual(test_y, predict_y)
	#模型的R方
	print("模型的R方是：%0.3f" %model.score(test_x, test_y))

#线性回归模型建立函数
def Linear_model_build(train_x, train_y):
	model = LinearRegression(normalize=opt.linear_normalize, fit_intercept=True)
	model.fit(train_x, train_y)
	return model

#多项式回归模型建立函数,二次函数拟合的时候:features=2
def polynomial_model_build(train_x, test_x, predict_x, train_y, features = opt.features):
	poly_features = PolynomialFeatures(features)
	poly_features.fit(train_x)
	#训练集的特征变换
	poly_train_x = poly_features.transform(train_x)
	#测试集的特征变换
	poly_test_x = poly_features.transform(test_x)
	#预测集的特征变换
	poly_predict_x = poly_features.transform(predict_x)
	poly_model = Linear_model_build(poly_train_x, train_y)
	return poly_model, poly_train_x, poly_test_x, poly_predict_x

#指数回归模型建立函数
def exp_model_build(train_x, test_x, predict_x, train_y):
	#训练集的特征变换
	size_r,size_c = train_x.shape
	exp_train_x = np.zeros((size_r, size_c))
	for i in range(size_r):
		for j in range(size_c):
			exp_train_x[i,j] = math.exp(train_x[i,j])
	#测试集的特征变换
	size_r,size_c = test_x.shape
	exp_test_x = np.zeros((size_r,size_c))
	for i in range(size_r):
		for j in range(size_c):
			exp_test_x[i,j] = math.exp(test_x[i,j])
	#预测集的特征变换
	size_r,size_c = predict_x.shape
	exp_predict_x = np.zeros((size_r,size_c))
	for i in range(size_r):
		for j in range(size_c):
			exp_predict_x[i,j] = math.exp(predict_x[i,j])
	exp_model=Linear_model_build(exp_train_x,train_y)
	return exp_model,exp_train_x,exp_test_x,exp_predict_x

'''
#多项式回归测试函数
if __name__ == '__main__':
	train_x = np.random.rand(10,3)
	train_y = np.random.rand(10,1)
	test_x = np.random.rand(5,3)
	test_y = np.random.rand(5,1)
	predict_x = np.random.rand(5,3)
	poly_model, poly_train_x, poly_test_x, poly_predict_x = \ 
		polynomial_model_build(train_x,test_x,predict_x,train_y,2)
	model_test(poly_test_x,test_y,poly_model)
	print(poly_train_x)
	print(train_x)
	model_coef_view(poly_model)
若train_x=np.mat([[x1,x2,x3]]),features=2
则poly_train_x=np.mat([[1,x1,x2,x3,x1*x1,x1*x2,x1*x3,x2*x3,x3*x3]])
'''

'''
#指数回归测试函数
if __name__ == '__main__':
	train_x = np.random.rand(10,3)
	train_y = np.random.rand(10,1)
	test_x = np.random.rand(5,3)
	test_y = np.random.rand(5,1)
	predict_x = np.random.rand(5,3)
	exp_model, exp_train_x, exp_test_x, exp_predict_x = \ 
		exp_model_build(train_x,test_x,predict_x,train_y)
	model_test(exp_test_x,test_y,exp_model)
	print(exp_train_x)
	print(train_x)
	model_coef_view(exp_model)
'''	
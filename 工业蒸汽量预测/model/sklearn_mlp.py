import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from config import DefaultConfig

opt = DefaultConfig()

class sklearn_mlp():
	def __init__(self, solver = opt.sklearn_solver, alpha = opt.sklearn_alpha, 
				hidden_layer_sizes = opt.sklearn_hidden_layer_sizes, 
				random_state = opt.sklearn_random_state, activation = opt.sklearn_activation):
		self.mlp = MLPRegressor(solver = solver, alpha = alpha, 
			hidden_layer_sizes = hidden_layer_sizes, random_state = random_state, 
			activation=activation)

	def train(self, train_data, train_label, validation_data, validation_label):
		self.mlp.fit(train_data, train_label)
		loss = self.get_loss(validation_data, validation_label)
		print("loss = ",end = '\t')
		print(loss)

	def get_loss(self, validation_data, validation_label):
		predict_y = self.mlp.predict(validation_data)
		predict_y = predict_y.reshape([validation_data.shape[0], 1])
		loss = mean_squared_error(predict_y, validation_label)
		return loss

	def predict(self, test_x):
		predict_np = np.array(self.mlp.predict(test_x))
		predict_np = np.reshape(predict_np, [predict_np.shape[0],1])
		return predict_np
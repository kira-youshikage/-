import torch as pt
import numpy as np
import torch.utils.data
from config import DefaultConfig

opt = DefaultConfig()

class torch_mlp(pt.nn.Module):
	def __init__(self, train_data_nc, train_label_nc):
		super(torch_mlp, self).__init__()
		self.fc1 = pt.nn.Linear(train_data_nc, 47)
		self.fc2 = pt.nn.Linear(47, 11)
		self.fc3 = pt.nn.Linear(11, 10)
		self.fc4 = pt.nn.Linear(10, train_label_nc)

	def forward(self, x):
		x = pt.nn.functional.relu(self.fc1(x))
		x = pt.nn.functional.dropout(x, p = opt.torch_p_1, training = self.training)
		x = pt.nn.functional.relu(self.fc2(x))
		x = pt.nn.functional.dropout(x, p = opt.torch_p_2, training = self.training)
		x = pt.nn.functional.relu(self.fc3(x))
		x = pt.nn.functional.dropout(x, p = opt.torch_p_3, training = self.training)
		x = self.fc4(x)
		return x

def torch_train(train_data, train_label, validation_data, validation_label, 
		  optimizer = opt.torch_optimizer,learning_rate = opt.torch_learning_rate, 
		  momentum = opt.torch_moment, batch_size = opt.torch_batch_size, 
		  net = None, epoch_time = opt.torch_epoch_time):
	train_data_nc = train_data.shape[1]
	train_label_nc = train_label.shape[1]
	if net == None:
		net = torch_mlp(train_data_nc, train_label_nc)
	if optimizer == 'Momentum':
		optimizer = pt.optim.SGD(net.parameters(), 
								lr = learning_rate, momentum = momentum)
	elif optimizer == 'Adam':
		optimizer = pt.optim.Adam(net.parameters(), lr = learning_rate)
	elif optimizer == 'SGD':
		optimizer = pt.optim.SGD(net.parameters(), lr = learning_rate)
	else:
		print("优化器参数输入错误")
		return
	train_data = pt.autograd.Variable(pt.from_numpy(train_data))
	train_label = pt.autograd.Variable(pt.from_numpy(train_label))
	validation_data = pt.from_numpy(validation_data)
	validation_label = pt.from_numpy(validation_label)
	torch_dataset = torch.utils.data.TensorDataset(train_data, train_label)
	train_loader = torch.utils.data.DataLoader(torch_dataset,
		batch_size = batch_size, shuffle = True)
	Loss = pt.nn.MSELoss()
	for i in range(epoch_time):
		net.train()
		for batch_idx, (data, label) in enumerate(train_loader):
			data = data.reshape(-1, train_data_nc)
			predict = net(data)
			loss = Loss(predict.float(), label.float())
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		if i % opt.torch_time == 0:
			print(str(i) + "\t" + "epoch：")
			print("\t" + "train_loss：" + str(loss))
			predict = net(validation_data)
			loss = Loss(predict.float(), validation_label.float())
			print("\t" + "validation_loss：" + str(loss))
	return net

def torch_predict(test_data, net):
	test_data = pt.autograd.Variable(torch.from_numpy(test_data))
	predict = net(test_data)
	predict = predict.data.numpy()
	return predict



'''
包含了对txt的读取和写入功能
例如read_data('读取测试.txt')
write_data中的参数是矩阵
'''
"""
第一次修改内容：
    代码规范化
"""
import numpy as np

def read_data(ID):
	f = open(ID,"r")   #设置文件对象
	str_str = f.read()
	f.close() #关闭文件
	return str_str

#写入数据
#如果在写之前先读取一下文件，再进行写入，则写入的数据会添加到文件末尾而不会替换掉原先的文件。
#这是因为指针引起的，r+ 模式的指针默认是在文件的开头，如果直接写入，则会覆盖源文件，
#通过read() 读取文件后，指针会移到文件的末尾，再写入数据就不会有问题了。
def write_data(ID,data_mat):
	f2 = open(ID,'wt+')
	size_r,size_c = data_mat.shape
	for i in range(size_r):
		for j in range(size_c):
			f2.read()
			f2.write(str(data_mat[i,j]))
		f2.read()
		f2.write('\n')
	f2.close()


#方差分析
"""
第一次修改内容：
	代码规范化
第二次修改内容：
	数据名字通化
"""
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy import stats
import pandas as pd
import numpy as np
'''
def ANOVA(x):
	#获得数据矩阵的行数和列数
	size_r,size_c = x.shape
	#给每一列命名，从A开始
	name = []
	for i in range(size_c):
		a = 'A' + str(i)
		name.append(a)
	#原数据矩阵x转DataFrame，列索引是A,B,C……
	df = pd.DataFrame(x,columns = name)
	formula = name[size_c-1] + '~'
	for i in range(size_c - 2):
		formula =formula +'C('+ name[i] + ')+'
	formula = formula+'C(' + name[size_c - 2]+')'
	#print(formula)
	anova_results = anova_lm(ols(formula,df).fit())
	return anova_results
'''
def ANOVA(x):
	#获得数据矩阵的行数和列数
	size_r,size_c = x.shape
	#给每一列命名，从A开始
	name = []
	for i in range(size_c):
		a = 'A' + str(i)
		name.append(a)
	#原数据矩阵x转DataFrame，列索引是A,B,C……
	df = pd.DataFrame(x,columns = name)
	formula = name[size_c-1] + '~'
	for i in range(size_c - 2):
		formula =formula + name[i] + '+'
	formula = formula + name[size_c - 2]
	#print(formula)
	anova_results = anova_lm(ols(formula,df).fit())
	return anova_results

'''
#以下为测试用
def test():
	x=np.mat([[1,2],[2,3],[2,2],[1,3],[1,5]])
	anova_results=ANOVA(x)

test()
'''

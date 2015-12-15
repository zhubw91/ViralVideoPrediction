import numpy as np
import scipy as sp
import math
import random
import glob,os
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures


def discrete(num):
	if num < 0.01:
		return 1
	elif num < 0.05:
		return 2
	elif num < 1:
		return 3
	elif num < 2:
		return 4
	elif num < 5:
		return 5
	elif num < 10:
		return 6
	elif num < 20:
		return 7
	elif num < 40:
		return 8
	elif num < 80:
		return 9
	else:
		return 10
def read_data(filelist):
	l = len(filelist)
	data = {}
	for filename in filelist:
		file_object = open('NewData/' + filename)
		for line in file_object:
			s = line.split()
			if len(s) == 0 or s[0] == 'Vid':
				continue
			if s[0] not in data:
				data[s[0]] = []
			data[s[0]].append(s[1])
	return data

filelist = ['data1.txt','data2.txt','data3.txt','data4.txt','data5.txt','data6.txt','datarandom1.txt','datarandom2.txt','datarandom3.txt','datarandom4.txt','datarandom5.txt','datarandom6.txt']
data = read_data(filelist)
output = open('traindata2.csv','w')
for key in data:
	if len(data[key]) == 6 and data[key][0] != '0':
		tmp = []
		for index,num in enumerate(data[key]):
			if index > 0:
				tmp.append(str( discrete (((int(data[key][index])+0.0)/(int(data[key][index-1]))-1)*100) ))
		output.write(','.join(tmp))
		output.write('\n')


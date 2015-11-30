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

filelist = ['data1.txt','data2.txt','data3.txt','data4.txt','data5.txt','data6.txt']
data = read_data(filelist)
output = open('traindata2.csv','w')
for key in data:
	if len(data[key]) == 6:
		tmp = []
		for index,num in enumerate(data[key]):
			if index > 0:
				tmp.append( str((int(data[key][index])+0.0)/int(data[key][index-1]) - 1) )
		output.write(','.join(tmp))
		output.write('\n')


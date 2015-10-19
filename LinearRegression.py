__author__ = 'Tingrui'


import numpy as np
import scipy as sp
import glob,os
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


def rmse(y_test,y):
    return sp.sqrt(sp.mean((y_test-y)**2))

def R22(y_test, y_true):
    y_mean = np.array(y_true)
    y_mean[:] = y_mean.mean()
    return 1 - rmse(y_test, y_true) / rmse(y_mean, y_true)

def get_file_list(dir_path,extension_list):
    os.chdir(dir_path)
    file_list = []
    for extension in extension_list:
        extension = '*.' + extension
        file_list += [os.path.realpath(e) for e in glob.glob(extension) ]
    return file_list

def get_train_array(filename):
    file_object=open(filename)
    x=[]
    y=[]
    for line in file_object:
        if line=='':
            break
        str=line.split('\t')
        if str[0]=='Date':
            continue
        tmp=[]
        tmp.append((int)(str[1]))
        x.append(tmp)
        y.append((int)(str[2]))
    return x,y

degree=[1,2,3,5]
y_test=[]
y_test=np.array(y_test)

dir_path = 'C:\CMUcourses\Capstone project\insight_prased_txt\prased_txt'
extension_list=['pattern']

a=get_file_list(dir_path,extension_list)
for file in a:
    x,y=get_train_array(file)
    print file
    for d in degree:
        clf = Pipeline([('poly', PolynomialFeatures(degree=d)),  ('linear', LinearRegression(fit_intercept=False))])
        clf.fit(x, y)
        y_test=clf.predict(x)
        #print R22(y_test,y)
        print clf.predict(x[-1]), x[-1]

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


def rmse(y_test,y):
    return sp.sqrt(sp.mean((y_test-y)**2))

def cal_map(list1, list2, n):
    a = [i[0] for i in sorted(enumerate(list1), key=lambda x:x[1])]
    b = [i[0] for i in sorted(enumerate(list2), key=lambda x:x[1])]
    result = 0
    for i in range(n):
        tmp = 0
        for j in range(i):
            if a[j] == b[i]:
                tmp += 1.0
        result = result + tmp / (i+1)

    result = result/n
    return result
        

def R22(y_test, y_true):
    y_mean = np.array(y_true)
    y_mean[:] = y_mean.mean()
    return 1 - rmse(y_test, y_true) / rmse(y_mean, y_true)

def get_file_list(dir_path,extension_list):
    os.chdir(dir_path)
    file_list = []
    for extension in extension_list:
        extension = '*.' + extension
        file_list += [os.path.realpath(e) for e in glob.glob(extension)]
    return file_list

# Ti: indicator time
# Tr: Reference time
# Use accumulate view count on Ti to prefict accumulate view count on Tr
# def get_train_array(filename, ti, tr):
#     file_object=open(filename)
#     x=-1
#     y=-1
#     count = 0
#     for line in file_object:
#         if line=='':
#             break
#         s=line.split('\t')
#         # Ignore the header
#         if s[0]=='Date':
#             continue
#         count += int(s[2])
#         # No data on such day, just ignore this video
#         if (int(s[1]) > ti and x == -1) or (int(s[1]) > tr and y == -1):
#             return -1,-1
#         elif int(s[1]) == ti:
#             x = count
#         elif int(s[1]) == tr:
#             y = count
#             break     
#     return x,y

def get_train_array(filename, ti, tr, low, high):
    file_object=open(filename)
    count = 0
    i = 0
    for line in file_object:
        if line=='':
            break
        s=line.split('\t')
        # Ignore the header
        if s[0]=='Date':
            continue
        count += int(s[2])
        # No data on such day, just ignore this video
        if i > tr and x == -1 :
            return -1,-1
        elif i == ti:
            x = count
        elif i == tr:
            if int(s[1]) >= low and int(s[1]) <= high:
                y = count
                break    
            else:
                return -1,-1
        i += 1 
    return x,y


def performRegression(file_name, ti):
    data_x_log = []
    data_y_log = []
    data_y = []
    file = open(file_name)
    for line in file:
        
        s = line.split(',')
        x = int(s[ti])
        y = int(s[-1])

        # Take Log or not
        if x <= 0:
            data_x_log.append([0])
        else:
            data_x_log.append([math.log(x)])
        if y <= 0:
            data_y_log.append(0)
        else:
            data_y_log.append(math.log(y))
        data_y.append(y)



    # Cross Validation
    fold_num = 10
    index_list = [x for x in range(len(data_y))]
    random.seed(23333)
    random.shuffle(index_list)
    test_size = int(len(data_y)/fold_num)

    result1 = []
    result5 = []
    result10 = []
    result20 = []
    for k in range(fold_num):
        train_x = [data_x_log[index_list[i]] for i in range(test_size*k)] + [data_x_log[index_list[i]] for i in range(test_size*(k+1),len(data_x_log))]
        train_y = [data_y_log[index_list[i]] for i in range(test_size*k)] + [data_y_log[index_list[i]] for i in range(test_size*(k+1),len(data_x_log))]
        test_x = [data_x_log[index_list[i]] for i in range(test_size*k,test_size*(k+1))]
        test_y = [data_y[index_list[i]] for i in range(test_size*k,test_size*(k+1))]
        clf = LinearRegression()
        #clf = LogisticRegression()
        clf.fit(train_x,train_y)
        predict_y_log = clf.predict(test_x)
        predict_y = [math.e**i for i in predict_y_log]
        predict_y = np.array(predict_y)
        test_y = np.array(test_y)

        result1.append(cal_map(test_y,predict_y,1))
        result5.append(cal_map(test_y,predict_y,5))
        result10.append(cal_map(test_y,predict_y,10))
        result20.append(cal_map(test_y,predict_y,20))
        #for i in range(len(test_y)):
        #    print test_y[i],predict_y[i]
    print sum(result1)/10
    print sum(result5)/10
    print sum(result10)/10
    print sum(result20)/10

performRegression("realdata_all.csv",12)





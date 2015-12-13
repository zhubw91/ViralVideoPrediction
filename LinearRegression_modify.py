import numpy as np
import scipy as sp
import glob,os
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

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

def get_cluster_list(filename):
    file_object = open(filename)
    result = {}
    for line in file_object:
        s = line.split()
        result[s[0]] = s[1]
    return result



def get_file_list(dir_path,extension_list):
    os.chdir(dir_path)
    file_list = []
    for extension in extension_list:
        extension = '*.' + extension
        file_list += [os.path.realpath(e) for e in glob.glob(extension)]
    return file_list

# n: number of data points we are going to use
# low,high: thresholds for the number of days on nth data point
def get_train_array(filename, n, low, high, cluster_list):
    filename_real = filename.split('/')[-1]
    if filename_real not in cluster_list:
        return []
    file_object = open(filename)
    result_y = []
    result_x = []
    count = 0
    index = 0
    for line in file_object:
        if line=='':
            break
        s=line.split('\t')
        # Ignore the header
        if s[0]=='Date':
            continue
        index += 1
        if index == n:
            if int(s[1]) >= low and int(s[1]) <= high:
                count += int(s[2])
                result_y.append(count)
                result_x.append([int(s[1])])
                return [result_x,result_y]
            else:
                return []

        count += int(s[2])
        result_y.append(count)
        result_x.append([int(s[1])])
    return []
        
def generate_data(file_list, cluster_list, d):

    y_test = []
    y_real = []

    for file in file_list:
        tmp = get_train_array(file, 20, 100, 200, cluster_list)
        if len(tmp) == 0:
            continue

        tmp_x,tmp_y = tmp
        clf = Pipeline([('poly', PolynomialFeatures(degree=d)),  ('linear', LinearRegression(fit_intercept=False))])
        clf.fit(tmp_x[:-1], tmp_y[:-1])
        y_test.append(clf.predict(tmp_x[-1])[0] - clf.predict(tmp_x[-2])[0])
        y_real.append(tmp_y[-1] - tmp_y[-2])
    print cal_map(y_real, y_test, 1),cal_map(y_real, y_test, 5),cal_map(y_real, y_test, 10),cal_map(y_real, y_test, 20)
            



dir_path = 'prased_txt'
extension_list=['pattern']
cluster_list = get_cluster_list('video1IndexMap.txt')
file_list=get_file_list(dir_path,extension_list)
generate_data(file_list, cluster_list, 3)


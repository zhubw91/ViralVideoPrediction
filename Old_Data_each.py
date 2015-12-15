import numpy as np
import scipy as sp
import math
import random
import glob,os


def get_cluster_list(filename):
    file_object = open(filename)
    result = {}
    for line in file_object:
        s = line.split()
        result[s[0]+'.pattern'] = s[1]
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
    file_object=open(filename)
    result = []
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
                count = int(s[2])
                result.append(count)
                result.append(cluster_list[filename_real])
                return result
            else:
                return []

        count += int(s[2])
        result.append(count)
    return []
        
        
def discrete(x):
    k = 10
    if x >= 500:
        return 10
    elif x <= 0:
        return 0
    else:
        delta = math.log(500) / k
        return int(round(math.log(x+1)/delta))


def generate_data(file_list, cluster, cluster_list):
    output = open('../traindata_each' + cluster + '.csv','w')
    output_real = open('../realdata_each' + cluster + '.csv','w')
    for file in file_list:
        tmp = get_train_array(file, 20, 100, 200, cluster_list)
        if len(tmp) == 0:
            continue
        if cluster != "all" and tmp[-1] != cluster:
            continue
        tmp = tmp[:-1]
        line = []
        for index,num in enumerate(tmp):
            if index > 0:
                if tmp[index-1] == 0:
                    line.append(discrete(tmp[index]))
                else:
                    line.append( discrete((((tmp[index]+0.0)/tmp[index-1]))*100) )
        line = [str(x) for x in line]

        output.write(','.join(line))
        output.write('\n')
        
        line = [str(x) for x in tmp]
        output_real.write(','.join(line))
        output_real.write('\n')


dir_path = 'prased_txt'
extension_list=['pattern']
cluster_list = get_cluster_list('pre_res.txt')
file_list=get_file_list(dir_path,extension_list)
generate_data(file_list,'all', cluster_list)


from sklearn import linear_model
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
import numpy
import scipy
import sklearn
import csv
import math
import random

# Train and Test
def SVM_predict(train_X, train_Y, test_X):
	clf = svm.SVC(C=1.0,gamma=0.1)
	clf.fit(train_X, train_Y)
	result = clf.predict(test_X)
	print "SVM Result:\n", result
        return result

def NB_predict(train_X, train_Y, test_X):
	clf = MultinomialNB()
	clf.fit(train_X, train_Y)
	result = clf.predict(test_X)
	print "NB Result:\n", result
        return result

def RandomForest_predict(train_X, train_Y, test_X):
	clf = RandomForestClassifier(n_estimators=10)
	clf.fit(train_X, train_Y)
	result = clf.predict(test_X)
	print "Random Forest Result:\n",result
        return result

#1 is standard, 2 is prediction
def calMAP(dict1,dict2,classnum):
    total=0
    pre=0.0
    for x in dict1:
        total+=1
        count=0
        if dict2.has_key(x):
            for y in dict1[x]:
                for z in dict2[x]:
                    if y==z:
                        count+=1
                        break
        pre+=(float)(count*1.0/len(dict1[x])*1.0)
    pre=pre/total*1.0
    print pre

label = []
feature = []
# num of levels for view counts
class_num = 10
train_set_ratio = 0.8

# Read CSV file
# Please check the viral.csv for names of header
with open("C:\CMUcourses\Capstone project\ViralVideoPrediction-master\\viral.csv","rb") as input_file:
	reader = csv.DictReader(input_file)
	# Store data in a list with the DictReader
	raw_data = []
	for line in reader:
		raw_data.append(line)

	# process the view count and generate lable list
	count_list = map(lambda x: int(x["view_count"]), raw_data)
	min_view_count, max_view_count = min(count_list), max(count_list)
	view_count_interval = (math.log(max_view_count) - math.log(min_view_count)) / class_num
	label = map(lambda x: int((math.log(x) - math.log(min_view_count)) / view_count_interval), count_list)

	# generate feature vector
	for line in raw_data:
		row = []
		for key in line:
			#print key
			if key in ['duration','num_raters','num_dislikes','num_likes']:
				#print key
				row.append(int(line[key]))
			elif key in ['avg_rate']:
				row.append(float(line[key]))
		feature.append(row)

# Generate training set and testing set
index_list = [x for x in range(len(label))]
random.seed(23333)
random.shuffle(index_list)
train_size = int(train_set_ratio*len(label))
train_x = [feature[index_list[i]] for i in range(train_size)]
train_y = [label[index_list[i]] for i in range(train_size)]
test_x = [feature[index_list[i]] for i in range(train_size,len(label))]
test_y = [label[index_list[i]] for i in range(train_size,len(label))]

svm_res=SVM_predict(train_x, train_y, test_x)
nb_res=NB_predict(train_x, train_y, test_x)
rf_res=RandomForest_predict(train_x, train_y, test_x)
svm_map={}
for i in range(10):
    for x in range(len(svm_res)):
        if svm_res[x]==i:
            if svm_map.has_key(i):
                svm_map[i].append(x)
            else:
                svm_map[i]=[]
                svm_map[i].append(x)
print svm_map

nb_map={}
for i in range(10):
    for x in range(len(nb_res)):
        if nb_res[x]==i:
            if nb_map.has_key(i):
                nb_map[i].append(x)
            else:
                nb_map[i]=[]
                nb_map[i].append(x)
print nb_map

rf_map={}
for i in range(10):
    for x in range(len(rf_res)):
        if rf_res[x]==i:
            if rf_map.has_key(i):
                rf_map[i].append(x)
            else:
                rf_map[i]=[]
                rf_map[i].append(x)
print rf_map

te_map={}
for i in range(10):
    for x in range(len(test_y)):
        if test_y[x]==i:
            if te_map.has_key(i):
                te_map[i].append(x)
            else:
                te_map[i]=[]
                te_map[i].append(x)
print te_map
print "True labels:\n",test_y

calMAP(te_map,rf_map,class_num)
calMAP(svm_map,rf_map,class_num)
calMAP(nb_map,rf_map,class_num)
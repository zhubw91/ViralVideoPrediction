from sklearn import linear_model
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
import numpy
import scipy
import csv
import math
import random

# Train and Test
def SVM_predict(train_X, train_Y, test_X):
	clf = svm.SVC()
	clf.fit(train_X, train_Y)
	result = clf.predict(test_X)
	print "SVM Result:\n", result

def NB_predict(train_X, train_Y, test_X):
	clf = MultinomialNB()
	clf.fit(train_X, train_Y)
	result = clf.predict(test_X)
	print "NB Result:\n", result

def RandomForest_predict(train_X, train_Y, test_X):
	clf = RandomForestClassifier(n_estimators=10)
	clf.fit(train_X, train_Y)
	result = clf.predict(test_X)
	print "Random Forest Result:\n",result

label = []
feature = []
# num of levels for view counts
class_num = 10
train_set_ratio = 0.8

# Read CSV file
# Please check the viral.csv for names of header
with open("viral.csv","rb") as input_file:
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

SVM_predict(train_x, train_y, test_x)
NB_predict(train_x, train_y, test_x)
RandomForest_predict(train_x, train_y, test_x)
print "True labels:\n",test_y




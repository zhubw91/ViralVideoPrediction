from sklearn import linear_model
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
import numpy
import scipy
from scipy.stats.stats import pearsonr
import sklearn
import csv
import math
import random
from sets import Set

# Train and Test
def SVM_predict(train_X, train_Y, test_X):
	clf = svm.SVC(C=1.0,gamma=0.1)
	clf.fit(train_X, train_Y)
	result = clf.predict(test_X)
	#print "SVM Result:\n", result
	return result

def NB_predict(train_X, train_Y, test_X):
	clf = MultinomialNB()
	clf.fit(train_X, train_Y)
	result = clf.predict(test_X)
	#print "NB Result:\n", result
	return result

def RandomForest_predict(train_X, train_Y, test_X):
	clf = RandomForestClassifier(n_estimators=10)
	clf.fit(train_X, train_Y)
	result = clf.predict(test_X)
	#print "Random Forest Result:\n",result
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

def getStopWordList(filepath):
	stopWords = Set([])
	with open(filepath,'r') as stopWordFile:
		for line in stopWordFile:
			for word in line.split():
				stopWords.add(word)
	return stopWords


def getTermIndxDict(row_data,stopWords):
	termIndxMap = {}
	#record all terms and its frequency in description
	for line in raw_data:
		for key in line:
			if key in ['video_desp']:
				descript = line['video_desp']
				current_terms = Set([])
				for term in descript.split():
					if term not in current_terms:
						current_terms.add(term)
						if term not in stopWords:
							if term in termIndxMap:
								termIndxMap[term] += 1
							else:
								termIndxMap[term] = 1
	#index terms with frequency >= 2
	for key,value in termIndxMap.items():
		if value == 1:
			del termIndxMap[key]
	indx = 0
	for key,value in termIndxMap.iteritems():
		termIndxMap[key] = indx
		indx += 1
	return termIndxMap


label = []
feature = []
# num of levels for view counts
class_num = 10
fold_num = 10

# Read CSV file
# Please check the viral.csv for names of header

#file_path = "C:\CMUcourses\Capstone project\ViralVideoPrediction-master\\viral.csv"
file_path = "viral.csv"
stop_words_file_path = "stop-words_english_3_en.txt"
stopWords = getStopWordList(stop_words_file_path)

with open(file_path,"rb") as input_file:
	reader = csv.DictReader(input_file)
	# Store data in a list with the DictReader
	raw_data = []
	for line in reader:
		raw_data.append(line)
	# get term index dictionary in description field
	termIndxDict = getTermIndxDict(raw_data,stopWords)
	#map for tracking uploader
	uploaderIds = {}
	# process the view count and generate lable list
	count_list = map(lambda x: int(x["view_count"]), raw_data)
	min_view_count, max_view_count = min(count_list), max(count_list)
	view_count_interval = (math.log(max_view_count) - math.log(min_view_count)) / class_num
	label = map(lambda x: int((math.log(x) - math.log(min_view_count)) / view_count_interval), count_list)
	#list for tracking bag of words in description
	descriptionsTerms = [0 for i in range(len(termIndxDict))]
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

		# Jinsub's Features

		# 1) Title length vs. description length
		#if len(line['video_title']) < len(line['video_desp']):
		#	row.append(0)
		#else:
		#	row.append(1)

		# 2) Title contains numeric
<<<<<<< HEAD
		#if line['video_title'].isalpha():
		#	row.append(0)
		#else:
		#	row.append(1)
		
=======
		if line['video_title'].isalpha():
			row.append(0)
		else:
			row.append(1)
		# uploader feature
		if line['uploader_id'] in uploaderIds:
			row.append(uploaderIds[line['uploader_id']])
		else:
			uploaderIds[line['uploader_id']] = len(uploaderIds)
			row.append(uploaderIds[line['uploader_id']])

		descript = line['video_desp']
		for term in descript.split():
			if term in termIndxDict:
				descriptionsTerms[termIndxDict[term]] += 1
		row.extend(descriptionsTerms)

>>>>>>> origin/master
		feature.append(row)

#calculate PCC
for i in range(len(feature[0])-1):
	singleFeature = [x[i] for x in feature]
	(pcc,pvalue) = pearsonr(singleFeature, label)
	print 'PCC for ',i,' th Feature is ',pcc
	print '2-Tailed P-Value for ',i,' th Feature is ',pvalue
# Generate training set and testing set
# With Cross Validation

index_list = [x for x in range(len(label))]
random.seed(23333)
random.shuffle(index_list)
test_size = int(len(label)/fold_num)
for k in range(fold_num):
	train_x = [feature[index_list[i]] for i in range(test_size*k)] + [feature[index_list[i]] for i in range(test_size*(k+1),len(label))]
	train_y = [label[index_list[i]] for i in range(test_size*k)] + [label[index_list[i]] for i in range(test_size*(k+1),len(label))]
	test_x = [feature[index_list[i]] for i in range(test_size*k,test_size*(k+1))]
	test_y = [label[index_list[i]] for i in range(test_size*k,test_size*(k+1))]

	svm_res=SVM_predict(train_x, train_y, test_x)
	nb_res=NB_predict(train_x, train_y, test_x)
	rf_res=RandomForest_predict(train_x, train_y, test_x)
	svm_map={}

	print str(k) + "th test:"
	for i in range(10):
		for x in range(len(svm_res)):
			if svm_res[x]==i:
				if svm_map.has_key(i):
					svm_map[i].append(x)
				else:
					svm_map[i]=[]
					svm_map[i].append(x)
	#print svm_map

	nb_map={}
	for i in range(10):
		for x in range(len(nb_res)):
			if nb_res[x]==i:
				if nb_map.has_key(i):
					nb_map[i].append(x)
				else:
					nb_map[i]=[]
					nb_map[i].append(x)
	#print nb_map

	rf_map={}
	for i in range(10):
		for x in range(len(rf_res)):
			if rf_res[x]==i:
				if rf_map.has_key(i):
					rf_map[i].append(x)
				else:
					rf_map[i]=[]
					rf_map[i].append(x)
	#print rf_map

	te_map={}
	for i in range(10):
		for x in range(len(test_y)):
			if test_y[x]==i:
				if te_map.has_key(i):
					te_map[i].append(x)
				else:
					te_map[i]=[]
					te_map[i].append(x)
	#print te_map
	#print "True labels:\n",test_y

	print "RF:"
	calMAP(te_map,rf_map,class_num)
	print "SVM:"
	calMAP(te_map,svm_map,class_num)
	print "NB:"
	calMAP(te_map,nb_map,class_num)

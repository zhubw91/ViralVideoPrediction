import csv
import time
import os
import datetime
from datetime import date
import matplotlib.pyplot as plt
from sets import Set
import numpy as np
from sklearn.preprocessing import normalize

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

def getTrendVector(input_file_path):
    date_vector = []
    trend_vector = []
    with open(input_file_path,"rb") as input_file:
        reader = csv.reader(input_file,delimiter='\t', quotechar='|') #read in file
        #skip 5 rows
        for i in range(5):
            next(reader,None)
        for row in reader:
            if len(row) < 1:
                break
            (date_range,count) = row[0].split(',')
            days = date_range.split(' - ')
            end_date = ""
            if len(days) == 2:
                (start_date,end_date) = date_range.split(' - ')
            else:
                end_date = days[0]
            triple = end_date.split('-')
            year = "2000"
            month = "1"
            day = "1"
            if len(triple) == 3:
                (year,month,day) = triple
            elif len(triple) == 2:
                (year,month) = triple
            date_vector.append(date(int(year),int(month),int(day)))
            trend_vector.append(int(count))
    return (date_vector,trend_vector)

def getViewCountVector(input_file_path):
    with open(input_file_path,"r") as input_file:
        reader = csv.reader(input_file,delimiter="\t")
        data = list(reader)
        dates = [row[0] for row in data]
        column = [row[2] for row in data]
        view_vector = [int(val) for val in column[1:]]
        date_vector = []
        for a_date in dates[1:]:
            (month,day,year) = a_date.split('/')
            year = "20" + year
            #print date(int(year),int(month),int(day))
            date_vector.append(date(int(year),int(month),int(day)))
    return (date_vector,view_vector)

def getFileNames(folder):
    filenames = Set([])
    for fn in os.listdir(folder):
        if fn.endswith(".pattern"):
            # get View Column for current file/video as Feature Vector
                #save file name to indx mapping
                name = fn.split('.')[0]
                filenames.add(name)
    return filenames

def getVideoId2Title(file_path):
    id2name = {}
    with open(file_path,"rb") as input_file:
        reader = csv.DictReader(input_file) #read in file
        for line in reader:
            video_title = str(line['video_title'])
            video_id = str(line['video_id'])
            id2name[video_id] = video_title
    return id2name

view_count_folder = 'prased_txt/'
filenames = getFileNames(view_count_folder)
trend_folder = 'googleTrends/'
save_folder = 'view_vs_trend/'
viral_file = 'viral.csv'
id2title = getVideoId2Title(viral_file)
for fn in os.listdir(trend_folder):
    if fn.endswith(".csv"):
        filename = fn.split('.')[0]
        (x,y) = getTrendVector(trend_folder+filename+'.csv')
        if len(x) > 0 and len(x) == len(y):
            plt.figure()
            y = np.array(y)
            norm_y = y / np.linalg.norm(y)
            plt.plot(x,norm_y,label="google trends")
            if filename in filenames:
                (x,y) = getViewCountVector(view_count_folder+filename+".pattern")
                y = np.array(y)
                norm_y = y / np.linalg.norm(y)
                plt.plot(x,norm_y,label="youtube view counts")
            plt.legend()
            if filename in id2title:
                plt.title(id2title[filename])
            #plt.show()
            plt.savefig(save_folder+filename+'.png')


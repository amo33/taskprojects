import sys 
import os 
import csv , json
from unicodedata import category
from numpy import NAN, NaN
import pandas as pd
from zmq import NULL 
#카테고리 file_path 소재 스타일
column_names = ["카테고리","file_path","소재",'스타일']

csvfile = pd.read_csv("test2.csv", sep=' ')
csvfile.dropna(inplace= True, how="all")
print(csvfile[:30])
#csvfile_drop = csvfile.dropna(axis= "file_path", how="any")
#csvfile_drop.reset_index(drop=True)
label_lst = csvfile["file_path"]
print(len(label_lst))
category_lst = csvfile["스타일"]
count = 0
print(label_lst)
print(category_lst)
for i in range(len(label_lst)):
    #print(label_lst[i])
    #print(category_lst[i])
    if category_lst[i] == NaN:
        print(label_lst[i])
        continue
    with open('./dataset/labeldata/'+category_lst[i]+"/"+str(label_lst[i])+'.json') as json_file:
       
        data =json.load(json_file)
        box_data = {}
        key_data = list(data['데이터셋 정보']['데이터셋 상세설명']['라벨링'].keys())
        #print(data)
        key = []
        for i in range(1, len(key_data)):
            if data['데이터셋 정보']['데이터셋 상세설명']['라벨링'][key_data[i]] != [{}]:
                #print(data['데이터셋 정보']['데이터셋 상세설명']['폴리곤좌표'])
                if (data['데이터셋 정보']['데이터셋 상세설명']['폴리곤좌표'][key_data[i]] != [{}]): 
                    key.append(key_data[i])
                    box_data[key_data[i]] = []
                    box_data[key_data[i]].extend(data['데이터셋 정보']['데이터셋 상세설명']['폴리곤좌표'][key_data[i]])
            
        #print(box_data)
        if (box_data == {}):
            count += 1
print(count)

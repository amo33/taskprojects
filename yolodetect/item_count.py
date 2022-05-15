import json
from glob import glob 
import os

#shuffle_data = os.listdir('train/annos') 
# 아직 무작위 추첨은 못했습니다.

cloth_dict = {}
count_dict = {}
cloth_dict["long sleeve dress"] = []
cloth_dict["shorts"] = []
cloth_dict["vest"] = []
cloth_dict["vest dress"] = []
cloth_dict["sling dress"] = []
cloth_dict["skirt"] = []
cloth_dict["short sleeve top"] = []
cloth_dict["sling"] = []
cloth_dict["trousers"] = []
cloth_dict["short sleeve dress"] = []
cloth_dict["short sleeve outwear"] = []
cloth_dict["long sleeve outwear"] = []
cloth_dict["long sleeve top"] = []

count_dict["long sleeve dress"] = 1000
count_dict["shorts"] = 1000
count_dict["vest"] = 1000
count_dict["vest dress"] = 1000
count_dict["sling dress"] = 1000
count_dict["skirt"] = 1000
count_dict["short sleeve top"] = 1000
count_dict["sling"] = 1000
count_dict["trousers"] = 1000
count_dict["short sleeve dress"] =1000
count_dict["short sleeve outwear"] = 1000
count_dict["long sleeve outwear"] =  543 # 개수 매우 적음
count_dict["long sleeve top"] = 1000

json_lst= glob('train/annos/*.json')
print(len(json_lst))
sub_index = 0 # the index of ground truth instance

for num in range(len(json_lst)):
    json_name = json_lst[num]
    file_name = json_lst[num][12:-5]

    if (num>=0):

        with open(json_name, 'r') as f:
            temp = json.loads(f.read())
            keys = list(temp.keys())
            item_keys = []
            for i in keys:
                if 'item' in i:
                    item_keys.append(i)
            for key in item_keys:
                category = temp[key]["category_name"]
                if count_dict[category] > 0:
                
                    cloth_dict[category].append(file_name)
                    count_dict[category] -= 1

cloth_key = list(cloth_dict.keys())

stored_files = []
total_files = []


for item in cloth_key:
    print("Key:%s\tValue:%s"%(item,cloth_dict[item]))
    total_files.extend(cloth_dict[item])

for i in total_files:
    if i not in stored_files:
        stored_files.append(i)

print(stored_files)
#print(len(stored_files))
#mkdir labeled ---
               # | img
               # | annos 
img_file_source = 'train/img/'
img_file_destination = 'labeled/img/'
json_destination = 'labeled/annos/'
json_file_source = 'train/annos/'
for files in stored_files:
    json_name = files + '.json'
    img_name = files +'.jpg'
    #os.replace(img_file_source + img_name,  img_file_destination+ img_name)
    os.replace(json_file_source + json_name, json_destination + json_name)
print(len(json_lst))
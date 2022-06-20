from numpy import NaN
import pandas as pd 
import json 
import os 
from pandas import json_normalize 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
# 이코드는 단순히 100개씩 잘라서 필요한 데이터만 선별해서 저장 
file_exists = os.path.exists('test.csv')

if file_exists ==True:
    result_df = pd.read_csv("test.csv", sep=' ', engine='python')
else:
    result_df = pd.DataFrame(index=range(0,0),columns={'스타일','카테고리' , '소재', 'file_path' })
option_style = os.listdir("dataset/Training/라벨링데이터/") 
print(option_style)
for i in range(len(option_style)):
    if option_style[i] == '.DS_Store':
        del option_style[i]
        break

for j in range(len(option_style)):
    lstjson = os.listdir('dataset/Training/라벨링데이터/'+option_style[j])

    for i in range(100):
        with open ('dataset/Training/라벨링데이터/'+option_style[j]+'/'+lstjson[i], "r") as data:
            
            info = json.load(data)
        print(lstjson[i])
        df = pd.DataFrame(json_normalize(info['데이터셋 정보']))

        options = ['데이터셋 상세설명.라벨링.아우터','데이터셋 상세설명.라벨링.하의','데이터셋 상세설명.라벨링.원피스','데이터셋 상세설명.라벨링.상의']

        style = df['데이터셋 상세설명.라벨링.스타일']
        style_key = style[0][0].keys()
        file_path = str(df['파일 번호']).split()[1]
        temp2 = []
        print(file_path)
        for i in options:
            temp = df[i]
            print(temp)
            if temp[0][0] != {}:
                temp_df = []
                keys = df[i][0][0].keys()
                print(keys)
                if '스타일' in style_key:

                    temp_df.append(style[0][0]['스타일'])
                else:
                    temp_df.append(None)

                if '카테고리' in keys:

                    temp_df.append(df[i][0][0]['카테고리'])
                else:
                    temp_df.append(None)
                if '소재' in keys:

                    temp_df.append(df[i][0][0]['소재'])
                else:
                    temp_df.append(None)

                temp_df.append(file_path)
                print(temp_df)
                temp2.append(temp_df)
                one_result = pd.DataFrame(temp2,columns=['스타일', '카테고리','소재', 'file_path'])
                print(one_result)
                result_df = result_df.append(one_result) # 업데이트를 해줘야한다.
                print("-----------------")
result_df.to_csv("test.csv",sep =' ',index=False)


import matplotlib.pyplot as plt
from pathlib import Path
import docker
import sys
import re
import time
import pyspark 
from pyspark.sql.functions import monotonically_increasing_id 
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import concat, col, lit
import pyspark.sql.functions as F
from pyspark.sql.functions import split, col
from pyspark.ml.feature import HashingTF, IDF
from pyspark.ml.feature import Normalizer
from pyspark.sql.types import StructType,StructField,ArrayType, StringType, LongType
import json 
from pyspark.sql.types import DoubleType
import uuid
import glob

def check_element(df2, number, info):
    print(df2.printSchema())
    if number == 3:
        df.filter(
            ((df.ArrayFeature[0] == info[0]) &(df.ArrayFeature[1] == info[1]) & (df.ArrayFeature[2] == info[2]))
        ).show()
        df2 = df2.unionAll(df.filter(
            (df.ArrayFeature[0] == info[0]) &(df.ArrayFeature[1] == info[1]) & (df.ArrayFeature[2] == info[2])
        ))
    elif number == 2 or number == 1: 
        df2 = df2.unionAll(df.filter(
            (df.ArrayFeature[0] == info[0]) | (df.ArrayFeature[1] == info[1]) | (df.ArrayFeature[2] == info[2])
        ))
    return df2 
spark = SparkSession\
        .builder\
        .appName('Python Spark basic example')\
        .config('spark.some.config.option', 'some-value')\
        .getOrCreate()


spark.conf.set("spark.sql.autoBroadcastJoinThreshold", -1)
spark.conf.set("spark.python.worker.memory", '24g')

deptSchema = StructType([       
    StructField('스타일', StringType(), True),
    StructField('카테고리', StringType(), True),
    StructField('소재', StringType(), True),
    StructField('file_path', StringType(), True),
    StructField('id', LongType(), True)
])

shcema = StructType([
    StructField('스타일', StringType(), True),
    StructField('카테고리', StringType(), True),
    StructField('ArrayFeature', ArrayType(StringType()), True),
    StructField('file_path', StringType(), True),
    StructField('id', LongType(), True)
])

df = spark.read.json("test_with_style.json")
result_2 = [("모던",'팬츠',"우븐", None, df.collect()[-1][2]+1)] # 제나한테서 받아야하는건 앞에 3개
rdd = spark.sparkContext.parallelize(result_2)
df_spark = spark.createDataFrame(data = rdd, schema= deptSchema)
test_df = df_spark.withColumn('feature',
                            concat(col('스타일'),lit(" "), col("카테고리"), lit(" "), col("소재")))
df_spark = test_df.select(split(col("feature"), " ").alias("ArrayFeature"),"file_path","id",'카테고리','스타일')\
        .drop("feature",'소재')
emptyRDD = spark.sparkContext.emptyRDD() 
df2 = spark.createDataFrame(emptyRDD,shcema)
#print(df.select(F.col("ArrayFeature")[0]).show()) 
number = 3
while df_spark.count() < 10:
    df_spark=check_element(df_spark, number, result_2[0])
    number -= 1 
    if number <1:
        break 
df_spark.show()

''' 원본 of app.py
@app.route('/recommend', methods=['GET','POST'])
def recommendation():
    startTime = time.time()
    global df 
    result_2 = [("모던",'팬츠',"우븐", None, df.collect()[-1][2]+1)] # 제나한테서 받아야하는건 앞에 3개
    print(1)
    rdd = spark.sparkContext.parallelize(result_2)
    df_spark = spark.createDataFrame(data = rdd, schema= deptSchema)
    test_df = df_spark.withColumn('feature',
                                concat(col('스타일'),lit(" "), col("카테고리"), lit(" "), col("소재")))
    df_spark = test_df.select(split(col("feature"), " ").alias("ArrayFeature"),"file_path","id",'카테고리','스타일')\
            .drop("feature",'소재')
    print(2)
    
    combined_data = df.union(df_spark)
    print(3)
    hashingTF = HashingTF(inputCol="ArrayFeature", outputCol="tf")
    tf = hashingTF.transform(combined_data) # 맨 마지막 df row에 사용자 추가한 df 가 combined data
    #tf = hashingTF.transform(df) 테스트용 
    idf = IDF(inputCol="tf", outputCol="feature").fit(tf)
    tfidf = idf.transform(tf)

    normalizer = Normalizer(inputCol="feature", outputCol="norm")
    data = normalizer.transform(tfidf)

    dot_udf = F.udf(lambda x,y: float(x.dot(y)), DoubleType())
    val = result_2[0][4] # 고유 id로 추출 
    #val = df.collect()[0]['id'] 테스트용 
    result = data.alias("i").join(data.alias("j"), F.col("i.id") == val)\
        .select(
            F.col("i.id").alias("i"), 
            F.col("j.id").alias("j"), 
            F.col("j.file_path").alias("path"),
            F.col("j.카테고리").alias("category"),
            F.col("j.스타일").alias('style'),
            dot_udf("i.norm", "j.norm").alias("dot")).where(F.col("i") != F.col("j"))
    dot_array = [(row.path , row.category ,row.style,float(row.dot)) for row in result.collect()] # 카테고리 -> category로 명명 변경 
    print(5)
    recommended = sorted(dot_array, key=lambda x: x[2], reverse=True)    
    top_list = []
    top_category = ['니트웨어','셔츠','블라우스','베스트','점퍼','티셔츠','드레스','브라탑','후드티','가디건']
    for options in filter(lambda u: u[1] in top_category , recommended):
        #print(options)
        if os.path.exists("/docker_file/deepfashion/dataset/K-Fashion/labelprocess/"+ options[0]+".json"):
            top_list.append([options[0], options[2]])
            del top_category[top_category.index(options[1])] # 다른 카테고리도 골고루 뽑을 수 있게 
        if len(top_list) >= 5: # 단점 한 카테고리가 무더기로 나오는 현상이 발생  
            break
    #print(top_list) #확인용 
    bottom_list = []
    bottom_category = ['팬츠','스커트','청바지','레깅스','조거팬츠']
    for options in filter(lambda u: u[1] in bottom_category , recommended):
        #print(options)
        if os.path.exists("/docker_file/deepfashion/dataset/K-Fashion/labelprocess/"+ options[0]+".json"):

            bottom_list.append([options[0], options[2]])
            del bottom_category[bottom_category.index(options[1])] # 다른 카테고리도 골고루 뽑을 수 있게 
        if len(bottom_list) >= 5: # 단점 한 카테고리가 무더기로 나오는 현상이 발생  
            break
    #print(bottom_list) # 확인용
    candidate = []
    candidate.extend(top_list)
    candidate.extend(bottom_list)
    # top list = 상의 , bottom_list =하의 리스트 입니다. 
    # candidate는 추천 세트 후보들로 위에 2개 리스트에서 높은거부터 뽑아서 넣어줍니다.
    # !!!!! 이거 생각해보자. -> top list 와 bottom_list를 같이 합쳐서 진행해도 괜찮을까? 괜찮을거 같다. 어차피 골고루 뽑아서 재조합시키는게 나을거 같다.
    
    top_path ='static/top/' 
    bottom_path = 'static/bottom'
    result_path = 'static/recommendation' 
    for path_name in [top_path, bottom_path, result_path]: # 이전에 사용된 이미지들은 모두 삭제
        [os.remove(f) for f in glob.glob(path_name+'/*')]
   
    for i in range(len(candidate)):
        file_name = candidate[i][0]
        style = candidate[i][1]
        with open('/docker_file/deepfashion/dataset/K-Fashion/labelprocess/'+str(file_name)+'.json',encoding='utf-8-sig')  as json_file:
            data =json.load(json_file)
            box_data = {}
            territory = {}
            key_data = list(data['데이터셋 정보']['데이터셋 상세설명']['라벨링'].keys())

            key = []
            for i in range(1, len(key_data)):
                if data['데이터셋 정보']['데이터셋 상세설명']['라벨링'][key_data[i]] != [{}]:
                    #print(data['데이터셋 정보']['데이터셋 상세설명']['폴리곤좌표'])
                    if (data['데이터셋 정보']['데이터셋 상세설명']['폴리곤좌표'][key_data[i]] != [{}]): 
                        key.append(key_data[i])
                        box_data[key_data[i]] = []
                        box_data[key_data[i]].extend(data['데이터셋 정보']['데이터셋 상세설명']['폴리곤좌표'][key_data[i]][0]['좌표'])
                        territory[key_data[i]] = []
                        territory[key_data[i]].extend(data['데이터셋 정보']['데이터셋 상세설명']['렉트좌표'][key_data[i]])
            if (box_data != {}):
                mask_images(box_data,key, territory, file_name,style )
    

    
    top_lst = os.listdir(top_path)
    bottom_lst = os.listdir(bottom_path)

    result = get_recommendation_set(top_lst, bottom_lst) # 마지막 결과 출력 
    df = df.dropna(how='any')
    #df.drop(df.tail(1).index,inplace = True) #마지막 현 사용자에 대한 정보 없애기 
    #print(result)
    print("[DONE] print model1_time spent :{:.4f}".format(time.time() - startTime))
    if result != []:
        return render_template('recommendation.html',data = result)
    elif result is False:
        return render_template('recommendation.html',data = 'error')
#spark-3.2.1-bin-hadoop3.2.tgz
#/usr/bin/python3
#spark-3.1.1-bin-hadoop3.2.tgz
def get_recommendation_set(top_lst, bottom_lst):
    result_path = 'static/recommendation'
    top_path ='static/top/' # 경로 맞춰줘야한다.
    bottom_path = 'static/bottom'

    for top in top_lst:
        top_img = cv2.imread(top_path+'/'+top)
        for bottom in bottom_lst:
            #print(top)
            #print(bottom)
            bottom_img = cv2.imread(bottom_path+'/'+bottom)
            top_img = cv2.resize(top_img, (bottom_img.shape[1],top_img.shape[0]))
            result_test_concat = np.vstack([top_img, bottom_img])
            res =np.where(result_test_concat==False,255, result_test_concat)
            cv2.imwrite(result_path+"/"+str(uuid.uuid4())+".png", res)
    print(os.listdir(result_path)[:5])
    return os.listdir(result_path)[:5] # 실제 이미지 경로들을 보내줘야한다. 이미지는 앞에 root 경로로 static/recommendation

def mask_images(boxes,key,territory,file_name, style):      
    #마스킹 된 걸 한번에 저장하고 싶으면 이 위치에 넣어줘야한다.(밑에 두 줄 코드)
    #img = cv2.imread("1542.jpg")
    #im = np.zeros(img.shape, dtype = np.uint8)
    img = cv2.imread("/docker_file/deepfashion/dataset/K-Fashion/imagedata/"+str(style)+'/' + str(file_name) + ".jpg")
    for j in range(len(key)):
        dst2= []
        im = np.zeros(img.shape, dtype = np.uint8)
        channel_count = img.shape[2] 

        # aspect ration로 적은 정보를 가지고 있는 이미지는 제거 
        if territory[key[j]][0] == {}:
            continue
        h=np.int32( territory[key[j]][0]['세로'])
        w = np.int32( territory[key[j]][0]['가로'])
        if (float(w)/h < -1): # 이 수치는 deepfashion의 aspect ratio를 측정해 만든 것이다.
            continue
        
        l= (np.array(boxes[key[j]]).astype(np.int32))
        #print(l)
        y_low = np.int32(territory[key[j]][0]['Y좌표'])
        y_high = y_low +np.int32( territory[key[j]][0]['세로'])
        x_low = np.int32(territory[key[j]][0]['X좌표'])
        x_high = x_low + np.int32(territory[key[j]][0]['가로'])

        #croped = img[y_low:y_high, x_low:x_high].copy()

        #l = l - l.min(axis=0)

        mask = np.zeros(img.shape[:2], np.uint8)
        ctr = np.array(l).reshape((-1,1,2)).astype(np.int32)
        cv2.drawContours(mask, [ctr], -1, (255, 255, 255), -1, cv2.LINE_AA)

        ## (3) bit-operation 과정 
        dst = cv2.bitwise_and(img, img, mask=mask)

        ## (4) white background으로 변경 
        bg = np.ones_like(img, np.uint8)*255
        cv2.imwrite(str(j)+"test.png",mask)
        cv2.bitwise_not(bg,bg, mask=mask)
        dst2.append(bg + dst)
       
        result_img = dst2[0][y_low:y_high, x_low:x_high]
    '''


''' 
    #이건 바로 위에 있는 mask image 함수에서 잘못된 코드였는데 참고할 수 있으니 살려놓음
    for j in range(len(key)):
    
        im = np.zeros(img.shape, dtype = np.uint8)
        channel_count = img.shape[2] 

        # aspect ration로 적은 정보를 가지고 있는 이미지는 제거 
        if territory[key[j]][0] == {}:
            continue
        h=np.int32( territory[key[j]][0]['세로'])
        w = np.int32( territory[key[j]][0]['가로'])
        if (float(w)/h < 0.29): # 이 수치는 deepfashion의 aspect ratio를 측정해 만든 것이다.
            continue
        #box_loc = [0 for _ in range(len(x_y_keys))]
        #print(key[j])
        ignore_mask_color = (255,)*channel_count
        
        l= (np.array(boxes[key[j]]).astype(np.int32))
        #print(l)
        cv2.fillPoly(im, [l], ignore_mask_color)
        masked_img = cv2.bitwise_not(img, im) # bitwise 연산 

        y_low = np.int32(territory[key[j]][0]['Y좌표'])
        y_high = y_low +np.int32( territory[key[j]][0]['세로'])
        x_low = np.int32(territory[key[j]][0]['X좌표'])
        x_high = x_low + np.int32(territory[key[j]][0]['가로'])
        
        transparent = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
        transparent[:,:,0:3] = img
        im = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
        #print(im.shape)
        transparent[:, :, 3] = im
        transparent = transparent[y_low:y_high, x_low:x_high]
        #print(masked_img)
        if key[j] in top_category:
            #print(key[j])
            cv2.imwrite("static/top/result_test"+ file_name[:4] + str(j)+".png",transparent)
        else:
            #print(key[j])
            cv2.imwrite("static/bottom/result_test"+file_name[:4] + str(j)+".png",transparent)
    '''
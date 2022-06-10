# !pip install pyspark 

import pyspark 
from pyspark.sql.functions import monotonically_increasing_id 
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import concat, col, lit
import pyspark.sql.functions as F
from pyspark.sql.functions import split, col
from pyspark.ml.feature import HashingTF, IDF
from pyspark.ml.feature import Normalizer
import os 
from pyspark.sql.types import StructType,StructField,ArrayType, StringType, LongType
import json 
spark = SparkSession\
        .builder\
        .appName('Python Spark basic example')\
        .config('spark.some.config.option', 'some-value')\
        .getOrCreate()


deptSchema = StructType([       
    StructField('스타일', StringType(), True),
    StructField('카테고리', StringType(), True),
    StructField('소재', StringType(), True),
    StructField('file_path', StringType(), True),
    StructField('id', LongType(), True)
])

df = spark.read.json("test4.json")
#temp_df = temp_df.select("*").withColumn("id", monotonically_increasing_id())
# temp_df.show() 

result_2 = [("모던",'팬츠',"우븐", None, df.collect()[-1][2]+1)] # None 대신 저장 파일의 이름 및 경로가 저기 들어가야함
@app.route('/recommendation')
def recommendation():
    rdd = spark.sparkContext.parallelize(result_2)
    df_spark = spark.createDataFrame(data = rdd, schema= deptSchema)
    test_df = df_spark.withColumn('feature',
                                concat(col('스타일'),lit(" "), col("카테고리"), lit(" "), col("소재")))

    combined_data = df.union(df_spark)

    df_spark = test_df.select(split(col("feature"), " ").alias("ArrayFeature"),"file_path","id",'카테고리')\
            .drop("feature",'소재','스타일')

    hashingTF = HashingTF(inputCol="ArrayFeature", outputCol="tf")
    tf = hashingTF.transform(df)

    idf = IDF(inputCol="tf", outputCol="feature").fit(tf)
    tfidf = idf.transform(tf)

    normalizer = Normalizer(inputCol="feature", outputCol="norm")
    data = normalizer.transform(tfidf)

    from pyspark.sql.types import DoubleType
    dot_udf = F.udf(lambda x,y: float(x.dot(y)), DoubleType())
    val = result_2[0][4]
    data.alias("i").join(data.alias("j"), F.col("i.id") == val)\
        .select(
            F.col("i.id").alias("i"), 
            F.col("j.id").alias("j"), 
            dot_udf("i.norm", "j.norm").alias("dot")).where(F.col("i") != F.col("j"))

    dot_array = [(row.j  ,float(row.dot)) for row in result.collect()] 
    print(dot_array)

    recommended = sorted(dot_array, key=lambda x: x[1])    
    candidate = recommended[-5:] # candidate = 사용자에게 추천해줄 데이터들 
    def mask_images(boxes):

        for i in range(5):

            with open('train/' +  candidate[i] +'.json')  as json_file:
                data =json.load(json_file)
                box_data = {}
                key_data = list(data['데이터셋 정보']['데이터셋 상세설명']['라벨링'].keys())
                for i in range(1, len(key_data)):
                    if data['데이터셋 정보']['데이터셋 상세설명']['라벨링'][key_data[i]] != [{}]:
                        print(key_data[i])
                        box_data[key_data[i]] = []
                        box_data[key_data[i]].extend(data['데이터셋 정보']['데이터셋 상세설명']['폴리곤좌표'][key_data[i]])

        

#result.sort('dot',ascending=False).collect()[:6] # 너무 오래걸려 7초

# (line 90 대신 line 92 ~ 95) sorting 후 dot_array[][0]가 이미지 고유 저장 이름이므로 이거를 보내주면 된다. # 시간 소요 : 3초 
dot_array = [(row.j ,float(row.dot)) for row in result.collect()]
print(dot_array)
a = sorted(dot_array, key=lambda x: x[1])    
print(a[-5:][0]) # 이미지 이름들만 보내준다. 

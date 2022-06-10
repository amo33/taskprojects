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
from pyspark.sql.functions import rand

def shuffle_dataframe():
	global df_man
	global df_woman 
	df_man = df_man.orderBy(rand()) # shuffle 
	df_woman = df_woman.orderBy(rand()) 
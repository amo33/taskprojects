from random import shuffle
import tensorflow as tf 
import pandas as pd 
import numpy as np 
import pickle 
from sklearn.model_selection import train_test_split
import tensorflow_datasets as tfds 
df = pd.read_csv('winequality-red.csv',';')

def permutation_train_test_split(X, y, test_size=0.3, shuffle=True, random_state=1004):
    test_num = int(X.shape[0] * test_size)
    train_num = X.shape[0] - test_num

    if shuffle:
        np.random.seed(random_state)
        shuffled = np.random.permutation(X.shape[0])
        X = X.iloc[shuffled,:]
        y = y.iloc[shuffled]
        X_train = X.iloc[:train_num]
        X_test = X.iloc[train_num:]
        y_train = y.iloc[:train_num]
        y_test = y.iloc[train_num:]

    else:
        X_train = X.iloc[:train_num]
        X_test = X.iloc[train_num:]
        y_train = y.iloc[:train_num]
        y_test = y.iloc[train_num:]

    return X_train, X_test, y_train, y_test

tf.reset_default_graph()
white_df = pd.DataFrame(df)
white_df_duplicate_dropped = df.copy()
white_df_duplicate_dropped.drop_duplicates(subset=None, inplace=True)
y = white_df_duplicate_dropped[["quality"]]
x = white_df_duplicate_dropped[['sulphates','alcohol','citric acid',"volatile acidity"]]
scaling_value = {}

def Standardscaler(col,data):
    mean_val = np.mean(data)
    std_val = np.std(data)
    scaling_value[col]=[mean_val,std_val]

    return data.apply(lambda x: (x - mean_val)/(std_val))

def outlier_iqr(data):
    q1, q3 = np.percentile(data, [25,75])
    
    iqr = q3 - q1 
    lower_bound = q1 - (iqr*1.5)
    upper_bound = q3 + (iqr*1.5)
    
    return np.where((data > upper_bound)|(data < lower_bound))


pH_outlier_index = outlier_iqr(x['citric acid'])[0]
sulphates_outlier_index = outlier_iqr(x['sulphates'])[0]
alcohol_outlier_index = outlier_iqr(x['alcohol'])[0]
acidity_outlier_index = outlier_iqr(x['volatile acidity'])[0]
total_outlier_index = np.unique(np.concatenate((pH_outlier_index, sulphates_outlier_index, alcohol_outlier_index)), 0)
print(total_outlier_index)
Not_outlier = []
print(x.shape)
for i in x.index:
    if i not in total_outlier_index:
        Not_outlier.append(i)
x_clean = x.loc[Not_outlier]
x_clean = x_clean.reset_index(drop=True)
y_clean = y.loc[Not_outlier]
y_clean = y_clean.reset_index(drop=True)
print(y_clean.shape)
print(x_clean.shape)
x_clean['sulphates'] = Standardscaler('sulphates',x_clean['sulphates'])
x_clean['alcohol'] = Standardscaler('alcohol',x_clean['alcohol'])
x_clean['citric acid'] = Standardscaler('citric acid',x_clean['citric acid'])
x_clean['volatile acidity'] = Standardscaler('volatile acidity', x_clean['volatile acidity'])
X = tf.compat.v1.placeholder(tf.float32, shape=[None, 4], name= "input")
Y = tf.compat.v1.placeholder(tf.float32, shape = [None, 1], name="output")

W = tf.Variable(tf.compat.v1.random_normal([4,1]), name ="weight")
b = tf.Variable(tf.compat.v1.random_normal([1]), name="bias")

hypothesis = tf.matmul(X, W)+b 
cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = 1e-4)
train = optimizer.minimize(cost)
acc = tf.equal(tf.round(hypothesis), Y)
acc = tf.reduce_mean(tf.cast(acc, tf.float32))
sess = tf.compat.v1.Session()
init = tf.compat.v1.global_variables_initializer()
sess.run(init)

saver = tf.compat.v1.train.Saver()


for step in range(10001):
    x_train_, x_test,y_train_,y_test = permutation_train_test_split(x_clean,y_clean,test_size=0.3, shuffle=True ,random_state=100)
    x_train, x_valid, y_train, y_valid = permutation_train_test_split(x_train_, y_train_, test_size=0.1, shuffle=True) 
    #print(x_train.shape, x_valid.shape,x_test.shape,y_train.shape,y_valid.shape,y_test.shape) 
    is_correct = tf.equal(tf.round(hypothesis)+1, y_valid)
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    
    tmp, loss_val , y_val = sess.run([train,cost,hypothesis], feed_dict={X:x_train, Y:y_train})
    if step % 1000 == 0:
        y_valid_val = sess.run([accuracy], feed_dict={X:x_valid})
        print("==================================")
        print("#",step, "Cost: ",loss_val)
        print('# predict:',y_val[step%1000])
        print('# validation value', y_valid_val)
        print("==================================")
    if step == 10000:
        saver.save(sess, 'redwinemodel/train1',global_step=60000)        

print("실제값", y_test)
print("정확도: ",sess.run(accuracy*100, feed_dict={X:x_test, Y:y_test}))

sess.close()

with open('redwine.pickle','wb') as fw:
    pickle.dump(scaling_value, fw)


#https://jjeamin.github.io/posts/Checkpoint/ 그래프 따로 저장 
#https://blog.naver.com/PostView.nhn?blogId=rhrkdfus&logNo=221480101170 모델 저장 공부용 . graph랑 헷갈리면 안된다.
#https://brunch.co.kr/@synabreu/79 tensor 1.15 공부
#https://tgjeon.github.io/blog/tf/Linear%20Regression.html 
import tensorflow as tf 
import pandas as pd 
import numpy as np 
import pickle, os.path
import time 
from customfunction import *
while True:
    #wineoption = input("What kind of wine:")
    wineoption = 'red'
    if wineoption == 'white':
        df = pd.read_csv('data/train/winequality-white.csv',';')
        
        options = ['density','alcohol','residual sugar','volatile acidity','chlorides']
        
        break

    elif wineoption == 'red':
        df = pd.read_csv('data/train/winequality-red.csv',';')
        #option = list(df.columns)
        
        options = ['sulphates','alcohol','citric acid',"volatile acidity"]
        #options = option[0:-1]
        break 
    print("Type red or white")


tf.reset_default_graph()
wine_df = pd.DataFrame(df)
#drop_index = wine_df[(wine_df['quality'] == 5)|( wine_df['quality'] == 6)].index
#wine_df_clean = wine_df.drop(drop_index,labels=range(0, 1000), aixs=0 )
#print(wine_df_clean['quality'].value_counts())
#print(wine_df)
idx = round(wine_df.shape[0]*0.3)
df_test = wine_df.iloc[0:idx,:]
print(df_test)
wine_df.drop(wine_df.index[0:idx], inplace=True)
print(wine_df)
if os.path.exists('data/test/wine-red.csv') == False:
    f = open('data/test/wine-red.csv','w')
    f.close()
    df_test.to_csv("data/test/wine-red.csv", header=True, index= False) # header= column, index= 옆에 숫자 column생긴다.

wine_df_duplicate_dropped = wine_df.copy()
wine_df_duplicate_dropped.drop_duplicates(subset=None, inplace=True)
#y = wine_df_duplicate_dropped[["quality"]]
#x = wine_df_duplicate_dropped[options]
print(len(options))
x = wine_df[options]
y= wine_df[['quality']]
scaling_value = {}



pre_time = 0
x_clean  = x.copy()
y_clean  = y.copy()
for attr in options:
    x_clean[attr] = Standardscaler(scaling_value, attr, x_clean[attr])
'''
parsed_index = [0 for i in range(len(options))]
i = 0
for attr in options:
   parsed_index[i] = outlier_iqr(x_clean[attr])[0]
   i +=1
total_outlier_index = np.unique(np.concatenate((parsed_index)), 0)
print(x.shape)
Not_outlier = []

for i in x.index:
    if i not in total_outlier_index:
        Not_outlier.append(i)
x_clean = x.loc[Not_outlier]
x_clean = x_clean.reset_index(drop=True)
y_clean = y.loc[Not_outlier]
y_clean = y_clean.reset_index(drop=True)
print(x_clean.shape)
'''


size = len(options)
X = tf.compat.v1.placeholder(tf.float32, shape=[None, size], name= "input")
Y = tf.compat.v1.placeholder(tf.float32, shape = [None, 1], name="output")

W = tf.Variable(tf.compat.v1.random_normal([size,1]), name ="weight")
b = tf.Variable(tf.compat.v1.random_normal([1]), name="bias")
hypothesis = tf.matmul(X, W)+b # todo 선형회귀 

global_step = tf.Variable(0, trainable = False)
decay_steps = 1000 
decay_rate = 0.95
lr = tf.train.exponential_decay(1e-3, global_step, decay_steps, decay_rate, staircase=True)

cost = tf.reduce_mean(tf.square(hypothesis - Y)) # todo 처음에 csv 나누기 learning rate scheduler , optimizer 변경 , out
#optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = lr)
optimizer = tf.compat.v1.train.MomentumOptimizer(lr ,0.03 ,use_nesterov = True)
#optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr)
train = optimizer.minimize(cost)

sess = tf.compat.v1.Session()
init = tf.compat.v1.global_variables_initializer()
sess.run(init)

saver = tf.compat.v1.train.Saver([W, b])
op =round(x_clean.shape[0]*0.3)
for step in range(10001): #
    
    x_train, x_valid,y_train,y_valid = permutation_train_test_split(x_clean,y_clean,test_size=0.3, random_state=100)
    tmp, loss_val= sess.run([train,cost], feed_dict={X:x_train, Y:y_train})
    
    current_time = time.time() 
    
    if step % 1000 == 0:
        print("===============================")
        print("#",step, "Cost: ",loss_val)
        print("time :",current_time - pre_time)
        print((step+1)%op)
        print(op)
        expect_val = sess.run([hypothesis], feed_dict={X:x_valid.iloc[step % op ].values.reshape(-1,4)})
        print("train value:",expect_val)
        print("Real value:",y_valid.iloc[step % op])
    pre_time = current_time 
    if step == 10000:
        saver.save(sess, wineoption+'winemodel/train1',global_step=60000)  
         


print("예측값", sess.run(hypothesis, feed_dict={X:x_valid}))
print("실제값", y_valid)
is_correct = tf.equal(tf.round(hypothesis), y_valid)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print("정확도: ",sess.run(accuracy*100, feed_dict={X:x_valid, Y:y_valid}))
print(wineoption)

sess.close()

with open(wineoption+'wine.pickle','wb') as fw:
    pickle.dump(scaling_value, fw)

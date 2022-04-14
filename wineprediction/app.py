import tensorflow as tf 
from flask import Flask, render_template, request,redirect, url_for, session
import pickle
import numpy as np
import pandas as pd
# session use to constantly save form data
app = Flask(__name__)
#CORS(app)
app.config['SECRET_KEY'] = '1234'

@app.route('/')
def initial_page():
    return render_template('wineweb.html')

@app.route('/wine', methods=['POST'])
def wine_select():
    wine = request.form['wine'] #form 예외처리 -> 안 들어오면 에러로 다시 redirect 시캬주기 
    if wine is None:
        print("Need to pick wine option!")
        return render_template('wineweb.html',error = "Error")

    # 공통 입력값
    session['alcohol']= request.form['alcohol']
    session ['volatile acidity']= request.form['volatile acidity']

    if wine == 'red':
        
        session['citric acid'] = request.form['citric acid']
        session['sulphates'] = request.form['sulphates']
        
    elif wine == 'white':
        
        session['density'] = request.form['density']
        session['chlorides'] = request.form['chlorides']
        session['residual sugar'] = request.form['residual sugar']
    
    return redirect(url_for('wine_result', winedata = wine))
    
@app.route('/wine/<winedata>',methods=["GET","POST"])
def wine_result(winedata):
   
    tf.compat.v1.reset_default_graph()
    with tf.compat.v1.Session() as sess:
        if winedata == 'red':
            saver = tf.compat.v1.train.import_meta_graph('redwinemodel/train1-60000.meta')
            saver.restore(sess, tf.train.latest_checkpoint('redwinemodel/./'))
            df = pd.read_csv('data/test/wine-red.csv',',')
        elif winedata == 'white':
            saver = tf.compat.v1.train.import_meta_graph('whitewinemodel/train1-60000.meta')
            saver.restore(sess, tf.train.latest_checkpoint('whitewinemodel/./'))
            df = pd.read_csv('data/train/winequality-white.csv',',')
            print(df)
        with open(winedata+'wine.pickle', 'rb') as fr:
            wine_scale_loaded = pickle.load(fr)
            print(wine_scale_loaded)

        #attributes = list(wine_scale_loaded.keys())
        attributes =['sulphates','alcohol','citric acid',"volatile acidity"]
        data = []
        for col in attributes:
            attr = session[col]
            print(col, attr)
            data.append(Standardscaler(wine_scale_loaded,col,np.float32(attr)))

        data  = np.float32(data) #데이터는 numpy 형태로 
        data = data.reshape(1,-1) #한 column에 대한 값들이면 (-1,1) 한 row에 대한 값이라면 (1,-1)
        send = tf.add(tf.matmul(data,sess.run("weight:0")),sess.run("bias:0"))
        print(sess.run("weight:0"))
        
        df = pd.DataFrame(df)
        result = df[["quality"]]
        #x = df[0:-1]
        #temp_attribute = list(df.columns)
        #attributes = temp_attribute[0:-1]
        
        x = df[attributes]
        idx= df.shape[0]
        print(idx)
        res = 0
        val_distribution = [0 for _ in range(1, 11)]
        for i in range(idx):
            data = []
            for col in attributes:
                attr = x.iloc[i][col]
                data.append(Standardscaler(wine_scale_loaded,col,attr))
            data = np.float32(data)
            
            data = data.reshape(1,-1)
            
            hypothesis = tf.add(tf.matmul(data,sess.run("weight:0")),sess.run("bias:0")) 
            hypothesis_result = np.round(hypothesis.eval())
            if hypothesis_result== result.iloc[i].values: # add_1:0 값이라고 뜨는건 형태를 add_1:0의 값을 사용했다라고 표현. 이건 결과 값이 아니다. eval()을 써야 값이 나온다. in order to change tensor to scalar
                    #print(np.round(hypothesis.eval()))
                val_distribution[int(hypothesis_result)] += 1
                res +=1
        
            if winedata == 'white':

                if (np.round(hypothesis.eval()))== result.iloc[i].values: # add_1:0 값이라고 뜨는건 형태를 add_1:0의 값을 사용했다라고 표현. 이건 결과 값이 아니다. eval()을 써야 값이 나온다. in order to change tensor to scalar
                    print(np.round(hypothesis.eval()))
                    
                    res +=1
            elif winedata == 'red':
                if (np.round(hypothesis.eval()))== result.iloc[i].values: # add_1:0 값이라고 뜨는건 형태를 add_1:0의 값을 사용했다라고 표현. 이건 결과 값이 아니다. eval()을 써야 값이 나온다. in order to change tensor to scalar
                    print(np.round(hypothesis.eval()))
                    res +=1
                #if (np.trunc(hypothesis.eval())+1)== result.iloc[i].values: # add_1:0 값이라고 뜨는건 형태를 add_1:0의 값을 사용했다라고 표현. 이건 결과 값이 아니다. eval()을 써야 값이 나온다. in order to change tensor to scalar
                #    print(np.trunc(hypothesis.eval())+1)
                #    res +=1
        
        print(res, idx)
        print("=================")
        print("result of prediction list")
        print(val_distribution)
    
        val=np.round(send[0][0].eval())
        #val = 0
        return render_template('wineweb.html',data = val)

def Standardscaler(scaling_value,col,data):
    mean_val = scaling_value[col][0]
    std_val = scaling_value[col][1]
    
    return (data - mean_val)/(std_val)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)

#https://mangastorytelling.tistory.com/entry/K-ICT-%EB%B9%85%EB%8D%B0%EC%9D%B4%ED%84%B0%EC%84%BC%ED%84%B0-Ch8-%EC%99%80%EC%9D%B8-%ED%92%88%EC%A7%88-%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%A5%BC-%ED%99%9C%EC%9A%A9%ED%95%9C-%EB%B6%84%EC%84%9D-%EB%AA%A8%EB%8D%B8%EB%A7%81-%EC%84%A0%ED%98%95%ED%9A%8C%EA%B7%80%EB%AA%A8%EB%8D%B8-%EA%B7%9C%EC%A0%9C%EC%84%A0%ED%98%95%ED%9A%8C%EA%B7%80%EB%AA%A8%EB%8D%B8-%EC%9E%84%EC%A0%95%ED%99%98%EA%B5%90%EC%88%98

   
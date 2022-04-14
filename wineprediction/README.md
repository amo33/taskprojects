## BEcarful using OUTLIER DELETE 
 ##### I used outlier to drop that might interrupt training and prediction. But as I used the outlier after scaling, The prediction rate shrinked at the surprising rate. 
 ###### 1. Use outlier and find out how many data dropped. 
 ###### 2. Try to shrink it in an low rate. (ex. 95% over and 5% less).

 ### These codes below are my own savings in order to remember model calling in tensorflow 1.15
 ```python
        #this example is when I use pickle to save model(I didn't use this method)
        if winedata == 'red':
            model = pickle.load(open('models/Redwine.pkl','rb')) #pickle 사용을 추천합니다 instead of sklearn.joblib
            # but this time, I used ckpt for tensorflow 1.15 (please use ckpt to save model)
        elif winedata == 'white':
            model = pickle.load(open('models/Whitewine.pkl','rb'))
            
        #prediction = model.predict(data) -> data 활용해서 구하면 된다.
        #print(prediction)
        #print(model.best_estimator_.coef_) 
        size= 3
        X = tf.placeholder(tf.float32, shape=[None, size])
        W = tf.Variable(tf.random_normal([size, 1]), name='weight')
        b = tf.Variable(tf.random_normal([1]), name='bias')
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        hypothesis = tf.add(tf.matmul(X,W),b)
        prediction = tf.round(hypothesis)
        #If I use this method after calling the model, there is an error on using weight and bias.(It doesn't call the right trained one.) 
```

#### This code is using Sklearn Scaler and GRIDSEARCHCV. Because of using GRIDCV it was too slow. (So I didn't use it.)
```python

        standard_scaler_x = preprocessing.StandardScaler()
        standard_scaler_y = preprocessing.StandardScaler()
        red_df = pd.DataFrame(df)
        red_df_duplicate_dropped = df.copy()
        red_df_duplicate_dropped.drop_duplicates(subset=None, inplace=True)
        y = red_df_duplicate_dropped[["quality"]]
        X = red_df_duplicate_dropped[['sulphates','alcohol','pH']]
        #X = standard_scaler_x.fit_transform(X)
        #y = standard_scaler_y.fit_transform(y)
        red_df_x_train, red_df_x_test,red_df_y_train, red_df_y_test = train_test_split(X,y,test_size=0.3, random_state=100)

        def svc_param_selection(X,y,nfolds):
            svm_parameters = [ {'kernel' : ['linear'], 
            'gamma' : [0.00001, 0.0001, 0.001, 0.01, 0.1, 1], 
            'C' : [0.01, 0.1, 1, 10, 100, 1000]} ] 
            #사이킷런에서 제공하는 GridSearchCV를 사용해 최적의 파라미터를 구함 
            clf = GridSearchCV(svm.SVR(), svm_parameters, cv=nfolds) 
            clf.fit(X, y) 
            print(clf.best_params_) #최고 점수
            return clf


        model = svm.SVR(kernel='linear') #,validation_split = 0.33 , callbacks=[model_callback]
        for i in range(10):
        start = i*139
        end = start + 139
        model.fit(red_df_x_train.loc[[i not in range(start,end)], :], red_df_y_train)
        model.fit(red_df_x_train, red_df_x_test) # 왜 cross validation 구현을 못하겠지?

        model = svc_param_selection(red_df_x_train, red_df_y_train.values.ravel(),5)
                                                            # scale안하면 dataframe 상태여서 values붙여야한다.

        if os.path.isfile('models/Redwine.pkl') == False:
            os.makedirs('models/Redwine.pkl')

        with open('models/Redwine.pkl','wb') as f: 
            pickle.dump(model, f)

        #print('training accuracy:', model.score(red_df_x_train, red_df_y_train))

        def my_score(result, answer):
            comparison = pd.DataFrame(answer)
            
            comparison['prediction'] = result
            comparison = round(comparison)
            evaluation = (comparison['quality'] == comparison['prediction'])
            success = (evaluation == True).sum()
            failure = (evaluation == False).sum()
            
            return success / (success+failure)

        #print('(category) train set accuracy', my_score(red_df_x_train, red_df_y_train))
        #print('(category) test set accuracy', my_score(red_df_x_test, red_df_y_test))
```

##### For Tensorflow 1.x - We use sess(tf.session 's) to train. And we find out the predcition rate. 
##### This example, I used test data for model's accuarcy but, normally we use validation data to prevent overfitting.
```python 

        for i in range(len(x_train)):
            h_val = sess.run(hypothesis, feed_dict={X: x_train.iloc[i,:].reshape(1,)})
            print("Answer:", y_train[i], " Prediction:", h_val)

        cost_val, acc_val = sess.run([cost, acc], feed_dict={X:x_train, Y:y_train})
        print("Cost:", cost)
        print("Acc:",acc_val)

        print("===========prediction===========")
        for i in range(x_test.shape[0]):
            pre = sess.run(hypothesis, feed_dict={X: x_test.iloc[i].reshape(1,)})
            print(x_test[i],"=>",pre[0])
```
```python
        # url 내로의 redirect시, front에서 받은 form action data를 redirect된 곳에서도 사용하게 하려면
        # Don't use json dump
        # EX) wine = json.dumps({"alcohol":alcohol,"pH":pH,"sulphates":sulphates})  redirect ('@@@', data = wine) (X)
        #Use session to hold data
```

```python 
        # tensorflow 1.15 -> 
        # if bring graph like this, I will get an stored weight structure and bias structure with random values.
        # I think I really didn't figure it out of how to use graph as saving and calling models.
        # Need to improve and find this method.\n So if you are not familiar with using graph, I recommend you to call value of stored weight and bias by using sess.run("weight's defined name") something like this.
        graph = tf.compat.v1.get_default_graph() # 저장된 graph가져오기 
        sess.run(tf.global_variables_initializer())
        W = graph.get_tensor_by_name("weight:0") # 모델 가져오기 
    

        b = graph.get_tensor_by_name("bias:0") #모델 가져오기 
        
```

```python
        data=np.float32([x.iloc[i]['density'],x.iloc[i]['alcohol'],x.iloc[i]['residual sugar'], x.iloc[i]['volatile acidity'], x.iloc[i]['chlorides']]) #데이터는 numpy 형태로
        # 사실상 data에 넣어주는 값은  이차원 리스트를 float32로 바꿔준다.
        data[0] = Standardscaler('density', data[0])
        data[1] = Standardscaler('alcohol',data[1])
        data[2] = Standardscaler('residual sugar',data[2])
        data[3] = Standardscaler('volatile acidity', data[3])
        data[4] = Standardscaler('chlorides', data[4])
        #This is same with
        data = []
        for col in attributes:
            attr = x.iloc[i][col] # 이차원 리스트로 만들다.
            data.append(Standardscaler(col,np.float32(attr)))
        data = np.float32(data)  # 이차원 리스트 자체를 만들어서 넣어준다. 밑에 코드처럼 쓰는것이 간결하다.
```

#### 이번에 코드짜면서 한번더 알게된 사실 : html에서 쓰이는 javascript 함수는 5개의 함수가 존재하는데 1개의 함수(A)라도 에러가 난다면 html에서 onclick에 쓰이는 함수(B)를 불러오는데 에러가 난다.
#### referecningError -> function B is not defined. 
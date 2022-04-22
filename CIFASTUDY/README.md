# Version 1(keras version)
- In keras version, I only implemented the model structure and train (no test is done)
## Installation

```conda 
    conda install -c anaconda keras
```

## Bayesian optimization 
### 1. surrogate model(근사수학모델) 
        surrogate model is an engineering method used when an outcome of interest can't be eacsily measured or computed. 
        다른 말로 한다면, if doing some expensive experiment we try to simulate. And at that simulation, the experiment trying to find the optimal value or answer might be done by doing multiple times. At this situation, we use surrogate model to find the optimal solution.  
### 2. 목적 함수 (black-box function)
        탐색 대상 함수 라고도 불린다고 한다. 어느 입력값 (x)에 대해 미지의 목적 함수(f(x))를 상정하고, f(x)를 최대로 만드는 최적해를 찾는것이 목표이다. 이때 이 함수와 하이퍼파라미터가 쌍으로 surrogate model을 만든다.
        해당 작성자는 목적 함수에 모델을 학습하고 test_loss의 값을 리턴하는 함수를 구현해서 이 test_loss를 줄이는 방향으로 최적의 해를 만드려고 한다. 

## Layer class : 네트워크에서의 레이어의 추상화된 클래스 
### KERAS CAN"T DO 
    1. gradients
    2. device placements
    3. 분산 학습
    4. N개 샘플의 텐서로 시작하지 않는 것
    5. 타입 체크 

### Things I store to get used to.
```python
    return cifar_train_data, cifar_train_filenames, cifar_train_labels, \
        cifar_test_data, cifar_test_filenames, cifar_test_labels, cifar_label_names
    # In this way, \ helps us to understand that the above code is the same as cifar_train_data, cifar_train_filenames, cifar_train_labels, cifar_test_data, cifar_test_filenames, cifar_test_labels, cifar_label_names("줄바꿈으로 인식 - 너무 길어서 임의로 보기 쉽게 \ 로 표현")
```

### Dense layer의 역할 
    Dense layer는 입력과 출력을 연결해주는 역할을 한다. 또한 입력과 출력에 대한 가중치를 모두 저장하고 있다. 
    가장 기본적인 머신러닝의 층으로 서로 연속적으로 상관관계가 있는 데이터가 아니라면 이 층을 통해 학습시킬 수 있는 데이터가 많다고 한다.

# Version 2(pytorch version)
- Pytorch is used and in this below, basic pytorch elements will be briefly summarized. 

### Arithmetic operations 

```python
    # matrix multiplication = matmul, y1,y2,y3의 값은 모두 같다.
    y1 = tensor @ tensor.T 
    y2 = tensor.matmul(tensor.T)
    y3 = tensor.rand_like(tensor)

    torch.matmul(tensor, tensor.T, out=y3) #전치 곱의 결과를  y3에 저장 

    # element-wise product(요소별 곱) y1, y2, y3 모두 같다.
    z1 = tensor * tensor
    z2 = tensor.mul(tensor)

    z3 = torch.rand_like(tensor)
    torch.mul(tensor, tensor, out=z3)

    # .item() method 
    # var1.item()같은 값은 tensor안의 값을 뽑아서 python의 값으로 추출한다.
```

### eval 함수 
nn.Module에서 train time과 eval time에서 수행하는 다른 작업을 수행할 수 있도록 switching 하는 함수
```python
    <modelname>.eval()
    #Dropout Layer , BatchNorm Layer 같은 곳에서 train time과 evaltime을 분리해서 생각해줘야한다.
    # eval 함수는 with_no grad 와 같이 쓰인다. 
    # eval 작업이 끝나면 <modelname>.train()을 실행하면 학습 상태로 바뀐다.
```

### rollaxis 
numpy.roll을 이용해 axis를 roll 하는 방법
```python
    #arr= [1,2,3,4] 
    rollaxis(arr, 3, 1) # parameter explanation :
                                                # arr = 실제 axis차원을 나타내는 numpy.ndarray
                                                # 3 = arr[3]을 움직인다.
                                                # 1 = arr[1] 인덱스까지 이동 
                                                # result -> [1,4,2,3]
                                                # 맨 왼쪽부터 0,1,2,3.. 인덱스이다.
```
### NN 주의점 
Neural net이 working하지 않는 37가지 이유 - 타 블로그에서 가져온 자료입니다.
원문: https://blog.slavv.com/37-reasons-why-your-neural-network-is-not-working-4020854bd607

점검 리스트등으로 활용 가능해 보여서 일부(이긴 한데 거의 전부)를 발췌하여 옮겨둔다.


만일 학습이 잘 안된다면,

먼저 해볼 것
동작한다고 알려진 모델을 가져다가 해본다. 평이한 loss를 쓰면 더 좋다
regularization, data augmentation 등 모두 off
finetuning하고 있다면, preprocessing을 다시한번 체크하라. 원래 모델에게 주어진 preprocessing 그대로 해야 한다.
input data가 correct한지 확인
적은 데이터셋(2~20샘플)으로 시작하라. 그 데이터셋에 overfit시킨 다음 조금씩 데이터를 더해보라.
위에서 빼놓은 것들을 더해간다. regularization, data augmentation, custom loss, complex models, etc.
그래도 안되면 다음을 점검하라

I. Dataset issues
dimension을 뒤바꾸거나, 모두 0으로만 이루어진 벡터를 넣고 있거나, 같은 배치만 반복해서 넣거나 하는 어이없는 실수들이 많으니 하나하나 찍어보면서 점검할것.
랜덤 데이터를 넣어보고 에러의 변화를 살펴보라. 만일 비슷하다면 net의 중간 어디선가 데이터가 (모두 0이 된다든지 하는 식으로) garbage로 변하고 있다.
입력 몇개만 가지고 생성되는 label과, 그 입력을 shuffle해보고 생성되는 label이 같은지 점검해볼것
올바른 문제를 풀고 있는건가 다시 점검(주식데이터 같은건 원래 랜덤이다. 패턴이 있을리 없다)
데이터 자체가 너무 더러울 수 있다. noise가 너무 많다거나, mis-label이 너무 많다거나 하는 문제들. 일일이 눈으로 확인해보는 수밖에 없다.
shuffle 꼭 할것. ordered data가 들어가면 학습이 잘 안된다.
class imbalance 문제 점검. 참고할 글
트레이닝셋은 충분한가. finetuning말고 scratch부터 하려면 많은 데이터가 필요하다.
batch안에 최대한 많은 label이 들어가도록.
batch size를 줄여라. batch size가 너무 크면 generalization능력을 떨어트리는 것으로 알려져 있다. 참고논문 arXiv:1609.04836
II. Data Normalization/Augmentation
정규화 할것
정규화 하고 training/validation/test set을 나누는 것이 아니고, 먼저 나눈 후 training set에 대해 평균, 분산을 구해 정규화하는 것이다.
data augmentation 너무 많이 하면 underfit한다.
pretrained model쓸 때는 입력에 항상 주의할것. 예) 구간이 [0, 1], [-1, 1], [0, 255]중 어느것인가
III. Implementation issues
좀 더 간단한 문제부터 풀어보라. 예를들어, 객체의 종류와 위치를 맞추는 것이라면, 일단 종류만 맞춰보라
우연히 맞을 확률 점검. 예를들어, 10개의 클래스를 맞추는 문제에서 우연히 맞을 negative log loss는 −ln(0.1)=2.302다.
loss function을 만들어 쓰고 있다면, 해당 loss가 잘 동작하는지 일일이 확인할 필요가 있다.
라이브러리가 제공한는 loss를 쓴다면, 해당 함수가 어떤 형식의 input을 받는지 명확히 확인할것. 예를들어, PyTorch에서, NLLLoss와 CrossEntropyLoss는 다른 형식의 입력을 받는다.
loss가 작은 term들의 합이라면, 각 term의 scale을 조정해야 할 수도 있다.
loss말고 accuracy를 써야 할 경우도 있다. metric을 loss로 잡는 것이 적절한지 다시 생각해볼 것.
net을 스스로 만들었다면
하나하나 제대로 동작하는지 확실히 하고 넘어가라
학습중 frozen layer가 있는지 점검해볼것.
expressive power가 부족할 수 있다. network size를 늘려볼 것.
input이 (k,H,W)=(64,64,64) 이런식이면 중간에 잘 되는지 안되는지 보기가 애매하다. prime number로 구성하든지 해서 잘 동작하는지 확인해보라.
Gradient descent를 직접 만들었으면, 잘 동작하는지 확인하라. 다음을 참고하라 1 23
IV. Training issues
한개나 두개의 예를 넣어서 학습해보고 잘 되는지 확인하라
net초기화가 중요할 수 있다. Xavier나 He를 시도해보라
hyperparameter를 이리저리 바꿔본다
regularization(예: dropout, batch norm, weight/bias L2 reg.등)을 줄여본다. 이 강의에서는 (overfitting보다) underfitting을 먼저 제거하라고 한다.
loss가 줄고 있다면 더 기다려보라
Framework들은 mode(training/test)에 따라 Batch Norm, Dropout등이 다르게 동작한다.
학습과정을 시각화 하라.
각 layer의 activations, weights, updates를 monitor할 것. 변화량이 적어야 (약 1-e3정도는 돼야) 학습이 다 된 것이다.
Tensorboard나 Crayon을 써라
activation의 평균값이 0을 상회하는지 주시할것. Batch Norm이나 ELU를 써라.
weights, biases의 histogram은 gaussian인 것이 자연스럽다(LSTM은 그렇지 않다). 해당 값들이 inf로 발산하는지 주시해야 한다.
optimizer를 잘 쓰면 학습을 빠르게 할 수 있다. 각종 gradient에 관한 훌륭한 참고 글
Exploding / Vanishing gradients
gradient clipping해야할 수도 있다.
Deeplearning4j에 좋은 가이드라인이 나온다 : activation의 표준편차는 주로 0.5에서 2사이다. 이를 벗어나면 vanishing/exploding activation을 의심해봐야 한다.
Increase/Decrease Learning Rate : 현재 lr에서 0.1이나 10을 곱하면서 바꾸어볼것.
RNN을 학습할 때, NaN은 큰 문제
처음 100 iteration안에 NaN을 얻는다면, lr을 줄여본다.
0으로 나눌 때 뿐 아니라, log에 0이나 음수가 들어가서 나올 수 있다.
NaN을 다루는 Russell Stewart의 훌륭한 글이 있다.
layer by layer로 조사해보면서 NaN을 찾아야 할 수도 있다.

### 이미지의 shape과 transpose를 해야하는 상황 

이미지를 학습에 써야하는 상황에서 이미지 처리를 어떻게 해야하는지 감을 못 잡았었다. 
가령, 어떤 코드는 transpose(1,2,0)를 해주고 어떤 코드는 안해준다. 처음에는 reshape을 써도 되겠지 했지만, 선임 개발자 덕분에 reshape의 위험성을 알게 되었다. 
data reshape은 데이터의 틀을 강제로 바꿔주기 때문에 이미지의 원본 형태가 손상될 수 밖에 없다. 이때 원본의 데이터를 꼭 transpose를 해줘야하는줄 알았다. 
해당 내용을 조금 더 공부해보니, transpose의 목적은 PIL.Image.frommarray 와 같이 데이터를 pillow 라이브러리를 통해 이미지를 생성하려 할때 쓴다고 한다. 
pIL의 이미지는 ['width','height','channel] 형태다. pytorch에서 학습을 시킨다면 ['channel','width','height']이기 때문에 형태에 변화를 준다. 
해당 코드(dataprocess file)에서 
```python 
    valid_image = cifar_valid_data[idx].transpose(1,2,0)
    valid_image = valid_image.astype(np.uint8)
    valid_image = Image.fromarray(valid_image)
```
볼 수 있듯이, 데이터를 이미지로 변환시에 transpose해줘야한다.

PIL.Image 형태 -> tensor 형태로 변환하는 방법
PIL (numpy.ndarray) 형태는 H*W*C이고 scale도 [0~255]이다. 그런데 Tensor형태로 모델에 학습한다면 C*H*W 형태여야하기 때문에 유의해야한다. 
어떻게 하면 바꿀 수 있을까 고민하다가 구글링을 통해 너무나도 자주 썼던 기능에서 한번에 된다는 것을 알게 되었다. (추후 이미지 관련 작업을 할 경우 참고)
```python
    torchvision.transforms.ToTensor() #Image to Tensor 
    torchvision.transforms.ToPILImage() # Tensor to Image  
```
### tensor형태에서 차원을 늘리거나 줄이는 방법 
-> unsqueeze와 squeeze를 사용하자.
해당 문제는 dataloader에 1개의 데이터를 넣어줄때 활용했다. 실제 프론트에서 가져오는 이미지는 1개일때 dataloader에서 batch를 1로 잡아도 엉뚱한 값으로 (가령 [1,32,32] 와 같이 3차원을 자른다.) 가져온다. 
이때 unsqueeze를 사용해서 가져오는 데이터의 차원을 늘리거나 줄여서 문제를 해결했다.

```python 
        testimg_data = tf(testimg)
        testimg_data = testimg_data.unsqueeze(0)
        print(testimg_data.size())
        classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
        testimg_loader= DataLoader(testimg_data, batch_size=1, num_workers= 0, shuffle = False)
    
        
        with torch.no_grad():
            for data in testimg_loader:
            
                plt.imshow(to_pil_image(torch.squeeze(data,0)),cmap='gray')
                output = testnet(data)
            
            _, predicted = torch.max(output, 1)
            answer= []
            print(predicted)
            for i in range(len(imageset)):
                answer.append(classes[predicted[i]])
            data = answer
    
```
주말 공부 출처:
https://ddangjiwon.tistory.com/category/Backend/Internet
https://velog.io/@inyong_pang/Data-Structure-Hash-Table%ED%95%B4%EC%89%AC-%ED%85%8C%EC%9D%B4%EB%B8%94 
linear probing and sha-256 적용 
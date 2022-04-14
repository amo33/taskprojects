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

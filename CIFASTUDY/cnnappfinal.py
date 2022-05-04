'''
#http://krasserm.github.io/2018/03/19/gaussian-processes/
#http://krasserm.github.io/2018/03/21/bayesian-optimization/
#https://wooono.tistory.com/102
#https://nittaku.tistory.com/264 -> 발표 준비용 (pooling 사용하는 이유 )
#https://bskyvision.com/700
#https://github.com/deep-diver/CIFAR10-img-classification-tensorflow/blob/master/CIFAR10_image_classification.ipynb -deep diver
'''
from flask import Flask, render_template, request
import pickle
import numpy as np 
import pandas as pd
import uuid
from CIfar4 import ResNet9
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import PIL ,os 
from pathlib import Path 
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms.functional as F
from torchvision.transforms.functional import to_pil_image
from matplotlib import pyplot as plt
from torchvision.datasets import ImageFolder

import torchsummary

Imagefolder = os.path.join('static','image')
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = Imagefolder
app.config['SECRET_KEY'] = '1234'
PATH = './cifar_net_tf.pth'
PATH2 = './checkpoint.pt'
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
model_conv = torchvision.models.resnet18(pretrained= True)
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 10)

testnet = model_conv.to(DEVICE)

#device = torch.device('cpu') 
if DEVICE == 'cuda':
    testnet = testnet.cuda()

testnet.load_state_dict(torch.load(PATH, map_location=DEVICE))

for param in testnet.parameters():
    param.requires_grad = False
#print(torchsummary.summary(testnet,(3,32,32),device='cpu'))
#testnet.load_state_dict(torch.load(PATH))

testnet.eval() 

@app.route('/')
def initialPage():
    return render_template('imagepredict.html')

@app.route('/image', methods=['POST'])
def submitImage():
    imageset = []
    imgsrc = []
    
    if request.files['image'] != '':
        imageset.append(request.files['image'])
    else:
        print("Issue occured")
        data = "Error"
        return render_template('imagepredict.html', data = data)

    tf = transforms.ToTensor()
    transform_test = transforms.Compose([ 
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.49139968, 0.48215827 ,0.44653124),(0.24703233,0.24348505,0.26158768))
                                      ])
    imageset[0]=(PIL.Image.open(imageset[0]))
    #imageset[0] = imageset[0].resize((32,32))
    
    #이미지 경로 저장 
    filesource = os.path.join(app.config['UPLOAD_FOLDER'], str(uuid.uuid1())+'input.jpg')
    imageset[0].save(filesource)
    imgsrc.append(filesource)
    
    # 데이터 전처리 
    #image = np.array(imageset[0])
    #image = image.reshape((-1, 32*32*3)) - 이건 안해도 된다
    #image = (image) / 255 -이건 안해도 된다
    #image = image.reshape(3, 32 ,32)
       
    testimg = imageset[0]
    
    testimg = np.array(testimg, dtype=np.uint8) # dtype 변형 
    
    testimg_data = tf(testimg)
    testimg_data = testimg_data.unsqueeze(0) #0차원에 차원 추가 - 4차원으로 변형 

    classes = ('plane', 'car', 'bird', 'cat',
        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    #입력 데이터 dataloader에 넣기
    testimg_loader= DataLoader(testimg_data, batch_size=1, num_workers= 0, shuffle = False)
 
    with torch.no_grad():
        for data in testimg_loader:
            #data = data.to(DEVICE)
            output = testnet(data)
            print(output)
            _, predicted = torch.max(output, 1)
            print(predicted)

        answer= []
        for i in range(len(imageset)):
            answer.append(classes[predicted[i]])
        data = answer
    
    '''
    with torch.no_grad():
       
        output = testnet(testimg_data)
           
        _, predicted = torch.max(output, 1)
        answer= []
        print(predicted)
        for i in range(len(imageset)):
            answer.append(classes[predicted[i]])
        data = answer
    '''
    '''
    with open("testdata.pickle",'rb') as file:

        testset = []
        while True:
            try: 
                testdata = pickle.load(file)
                testdata = testdata.reshape(18000, 3, 32 ,32)
                #testdata = testdata.transpose(0,2,3,1)
                
                #testdata = np.rollaxis(testdata, 1,4)
            except EOFError:
                break
            #testset.extend(testdata)
        #print(testset)
        testset = np.array(testset,dtype=np.float32)
        print(testset.shape)
    with open("testlabel.pickle",'rb') as label:
        testlabel = []
        while True:
            try: 
                testdata2 = pickle.load(label)
            except EOFError:
                break
            #testlabel.extend(testdata2)
        #print(testset)
        testlabel = np.array(testdata2)
    testlabel = to_categorical(testlabel, 10)
    print("testlabel: ",testlabel.shape)
    #print(testlabel)s
    
    
    #testset= torch.from_numpy(testset).type(torch.float32)
    #print(testset)
    print(testdata[0].shape)
    print(testdata[1].shape)
    print(testdata[2].shape)
    print(tf(testdata[0].transpose(1,2,0)).shape)
    for i in range(testdata.shape[0]):
      
        #testdata[i] = np.rollaxis(testdata[i],0,2)
        #testdata[i] = testdata[i].transpose(2,1,0)
        img = np.array(testdata[i])
        img = img.reshape(-1, 32*32*3)
        img = (img) / 255
        img = img.reshape(32,32,3)
        testdata[i] = tf(img)
        #testdata[i] = tf(testdata[i].transpose(1,2,0))
        #print(testdata[i].shape)
      
        #plt.imshow(testdata[i], interpolation='nearest')
        
        
    
    testlabel_tensor = tf(testlabel)
    testdata_tensor = torch.from_numpy(testdata)
    testdata_tensor = testdata_tensor.type(torch.FloatTensor)
    testset_tensor = testdata_tensor
    #testset_tensor = tfnormal(testdata_tensor)
    print(testlabel_tensor.size())
    print(testset_tensor.size())
    testset_set = TensorDataset(testset_tensor, testlabel_tensor[0])
    '''
    #실제 test set 활용
    '''
    test_tensor = ImageFolder(root='data/test',transform = transform_test)
    test_loader = DataLoader(test_tensor, batch_size=4, num_workers= 0, shuffle=False)
    total = 0
    correct = 0

    with torch.no_grad():
        for set in test_loader:
            images, labels = set
          
            outputs = testnet(images)
            # 가장 높은 값을  갖는 분류(class)를 정답으로 선택하겠습니다
            _, predicted = torch.max(outputs, 1)
           
            total += labels.size(0)
            
            correct+=int(torch.sum(predicted==labels.data))
    print(total)
    print(correct)
    '''
    return render_template('imagepredict.html', data = data, imgdata = imgsrc)


if __name__ == "__main__":
    app.run(debug=True)
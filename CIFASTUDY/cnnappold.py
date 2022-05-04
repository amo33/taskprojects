#http://krasserm.github.io/2018/03/19/gaussian-processes/
#http://krasserm.github.io/2018/03/21/bayesian-optimization/
#https://wooono.tistory.com/102
#https://nittaku.tistory.com/264 -> 발표 준비용 (pooling 사용하는 이유 )
#https://bskyvision.com/700
#https://github.com/deep-diver/CIFAR10-img-classification-tensorflow/blob/master/CIFAR10_image_classification.ipynb -deep diver
from flask import Flask, render_template, request,redirect, url_for, session
import pickle
import numpy as np 
import pandas as pd
import uuid
from CIfar import CifarNet
import torch
import torchvision.transforms as transforms
import PIL ,os 
from pathlib import Path 
from torch.utils.data import TensorDataset, DataLoader
Imagefolder = os.path.join('static','image')
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = Imagefolder
app.config['SECRET_KEY'] = '1234'
PATH = './cifar_net.pth'

@app.route('/')
def initialPage():
    return render_template('imagepredict.html')

@app.route('/image', methods=['POST'])
def submitImage():
    imageset = []
    imgsrc = []
    '''
    if request.files['image'].filename !='' and request.files['image1'].filename !='' and request.files['image2'].filename !='' and request.files['image3'].filename !='':
        imageset.append(request.files['image'])
        imageset.append(request.files['image1'])
        imageset.append(request.files['image2'])
        imageset.append(request.files['image3'])
    '''
    if request.files['image'] != '':
        imageset.append(request.files['image'])
    else:
        print("Issue occured")
        data = "Error"
        return render_template('imagepredict.html', data = data)
    #if imageset != None:
    print("No issue")
    testnet = CifarNet()
    testnet.load_state_dict(torch.load(PATH))
    testnet.eval() 
    tf = transforms.ToTensor()
    for i in range(len(imageset)):
        imageset[i]=(PIL.Image.open(imageset[i]))
        imageset[i] = imageset[i].resize((32,32))
        filesource = os.path.join(app.config['UPLOAD_FOLDER'], str(uuid.uuid1())+'input.jpg')
        imageset[i].save(filesource)
        imgsrc.append(filesource)
        imageset[i] = tf(imageset[i])
    '''
        #image = PIL.Image.open(image)
        #image1 = PIL.Image.open(image1)
        #image2 = PIL.Image.open(image2)
        #image3 = PIL.Image.open(image3)
        #image = image.resize(( 32, 32))
        #image1 = image1.resize((32, 32))
        #image2 = image2.resize(( 32, 32))
        #image3 = image3.resize((32, 32))
        #testimg = tf(image)
        #testimg1 = tf(image1)
        #testimg2 = tf(image2)
        #testimg3 = tf(image3)
        '''
    testimg = imageset[0]
    '''
    for i in range(len(imageset)-1):
        testimg = np.vstack((testimg, imageset[i+1]))
    '''
    testimg = testimg.reshape((1, 3, 32, 32))
    testimg = testimg.permute(0,2,3,1).type(torch.float32)
    '''
    #testimg_tensor = torch.tensor(testimg)
    #print(testimg_tensor)
    
    #testimg = (testimg.astype('float32') -testimg.mean()) / testimg.std()
    #testimg = (testimg.astype('float32'))
    
    #testimg = testimg.type(torch.float32)
    #testimg_tensor = (testimg/255 -0.5 ) /0.5 여기서 255로 나눠주지 않아도 된다. 이유: 1이 최대값으로 구성되어 있기 때문이다. 정규화만 해주면 되기 때문에 mean=0.5, std:0.5로 구성해준다.
    '''
    testimg = (testimg- 0.5)/0.5 
    testimg_tensor = torch.tensor(testimg)
    testimg_loader= DataLoader(testimg_tensor, batch_size=1, num_workers= 0, shuffle = False)
    classes = ('plane', 'car', 'bird', 'cat',
        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
   
    with torch.no_grad():
        for data in testimg_loader:
            output = testnet(data)
            print(output)
        
        _, predicted = torch.max(output, 1)
        answer= []
        print(predicted)
        for i in range(len(imageset)):
            answer.append(classes[predicted[i]])
        data = answer
    
    return render_template('imagepredict.html', data = data, imgdata = imgsrc)


if __name__ == "__main__":
    app.run(debug=True)

'''
import torch 
import numpy as np 
from torch.utils.data import TensorDataset, DataLoader
import pickle 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
from CIfar import CifarNet
def unpickle(file):
    with open(file, 'rb') as f:
        data = pickle.load(f, encoding="bytes")
    return data

def to_categorical(y, num_classes):
    return np.eye(num_classes, dtype='uint8')[y]

def load_cifar_10_data(data_dir, negatives=False): #흩어져있는 batches들 합친다.
    meta_data_dict = unpickle(data_dir+"/batches.meta")
    cifar_label_names = meta_data_dict[b'label_names']
    cifar_label_names = np.array(cifar_label_names)
    
    cifar_train_data = None 
    cifar_train_filenames = []
    cifar_train_labels = []
    
    for i in range(1,7):
        print(i)
        if i == 6:
            cifar_train_data_dict = unpickle(data_dir + "/test_batch")
        else:    
            cifar_train_data_dict = unpickle(data_dir+"/data_batch_{}".format(i))
        
        if i==1:
            cifar_train_data = cifar_train_data_dict[b'data']
        else:
            cifar_train_data = np.vstack((cifar_train_data, cifar_train_data_dict[b'data'])) # 아래로 합쳐
        cifar_train_filenames += cifar_train_data_dict[b'filenames']
        cifar_train_labels += cifar_train_data_dict[b'labels']
    cifar_train_data = cifar_train_data.reshape((len(cifar_train_data), 3, 32, 32))
    if negatives: # 왜 이게 필요한걸까
        cifar_train_data = cifar_train_data.transpose(0,2,3,1).astype(np.float32)
    else:
        cifar_train_data = np.rollaxis(cifar_train_data, 1,4)
    cifar_train_filenames = np.array(cifar_train_filenames)
    cifar_train_labels = np.array(cifar_train_labels)
    
    size = 0.3 
    territory = int(np.round(len(cifar_train_data)*size))
    print(territory)
    
    cifar_test_data = cifar_train_data[0:territory]
    cifar_test_filenames = cifar_train_filenames[0:territory]
    cifar_test_labels = cifar_train_labels[0:territory]
    cifar_train_data = cifar_train_data[territory:]
    cifar_train_filenames = cifar_train_filenames[territory:]
    cifar_train_labels = cifar_train_labels[territory:]
    with open('testdata.pickle','wb') as f:
        pickle.dump(cifar_test_data,f)
    with open('testlabel.pickle','wb') as fw:
        pickle.dump(cifar_test_labels, fw)
    #cifar_test_data_dict = unpickle(data_dir + "/test_batch")
    #cifar_test_data = cifar_test_data_dict[b'data']
    #cifar_test_filenames = cifar_test_data_dict[b'filenames']
    #cifar_test_labels = cifar_test_data_dict[b'labels']

    #cifar_test_data = cifar_test_data.reshape((len(cifar_test_data), 3, 32, 32))
    #if negatives:
    #    cifar_test_data = cifar_test_data.transpose(0, 2, 3, 1).astype(np.float32)
    #else:
    #    cifar_test_data = np.rollaxis(cifar_test_data, 1, 4)
        
    #cifar_test_filenames = np.array(cifar_test_filenames)
    #cifar_test_labels = np.array(cifar_test_labels)
    
    x_train = (cifar_train_data.astype('float32')/255 -0.5) / 0.5
    #x_train = cifar_train_data
    #print(x_train)
    val_range = int(np.round(len(x_train) * 0.2))
    valid_set = x_train[0:val_range]
    train_set = x_train[val_range:]
    
    y_train = to_categorical(cifar_train_labels, 10)
    
    train_label_set = y_train[val_range:]
    val_label_set = y_train[0:val_range]
    
    x_test = (cifar_test_data.astype('float32')/255 -0.5) / 0.5
    print(x_test.shape)
    #x_test = cifar_test_data
    y_test = to_categorical(cifar_test_labels, 10)
    train_images_tensor = torch.tensor(train_set)
    print(train_images_tensor.shape)
    #train_images_tensor=transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))(train_images_tensor)
    valid_images_tensor = torch.tensor(valid_set)
    #transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))(valid_images_tensor)
    #train_set, val_set = torch.utils.data.random_split(train_images_tensor, [len(train_images_tensor)*0.8, len(train_images_tensor)*0.2])
    train_label_tensor = torch.tensor(train_label_set)
    valid_label_tensor = torch.tensor(val_label_set)
    #train_label_set, val_label_set = torch.utils.data.random_split(train_label_tensor, [len(train_label_tensor)*0.8, len(train_label_tensor)*0.2])

    train_tensor = TensorDataset(train_images_tensor, train_label_tensor)
    train_loader = DataLoader(train_tensor, batch_size=4, num_workers= 0 , shuffle = True)
    
    valid_tensor = TensorDataset(valid_images_tensor, valid_label_tensor)
    valid_loader = DataLoader(valid_tensor, batch_size=4, num_workers= 0, shuffle = False)
    
    test_images_tensor = torch.tensor(x_test)
    #transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))(test_images_tensor)
    test_label_tensor = torch.tensor(y_test)
    test_tensor = TensorDataset(test_images_tensor, test_label_tensor)
    test_loader = DataLoader(test_tensor, batch_size=4, num_workers = 0, shuffle = False)
    return train_loader, valid_loader, test_loader

net = CifarNet()

hypothesis = nn.CrossEntropyLoss()

optimizer = optim.SGD(net.parameters(), lr=0.09) #momentum=0.9 삭제
#scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer,
#                                        lr_lambda=lambda epoch: 0.95 ** epoch,
#                                        last_epoch=-1,
#                                        verbose=False)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)
train_loader, valid_loader, test_loader \
= load_cifar_10_data('cifar-10-batches')

check_loss_val = 10000

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
net.train()

best = net.state_dict()
for epoch in range(25):   # 데이터셋을 수차례 반복합니다.
    
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # [inputs, labels]의 목록인 data로부터 입력을 받은 후;
        inputs, labels = data
        # 변화도(Gradient) 매개변수를 0으로 만들고
        optimizer.zero_grad()

        # 순전파 + 역전파 + 최적화를 한 후
        outputs = net(inputs)
        #print(labels)
        labels = labels.to(torch.float)
        #labels = labels.softmax(dim=1)
        
        loss = hypothesis(outputs, labels)
        #print(loss)
        loss.backward()
        optimizer.step()
        
            # 통계를 출력합니다.
        #running_loss += loss.item()
        if check_loss_val > loss.item():
                check_loss_val = loss.item() 
                best =  net.state_dict()
                #print(best)
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {loss.item() / 2000:.3f}')
            running_loss = 0.0
    scheduler.step() # scheduler를 아래처럼 사용하면 learning rate 업데이트가 안된다.
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)      
    print("lr: ", optimizer.param_groups[0]['lr']) #--> lr 로그 찍는거 가능 

print('Finished Training')

PATH = './cifar_net.pth'
#torch.save(net.state_dict(), PATH)
torch.save(best,PATH)
dataiter = iter(valid_loader)
images, labels = dataiter.next()
tnet = CifarNet()
tnet.load_state_dict(torch.load(PATH))
outputs = tnet(images)
_, predicted = torch.max(outputs, 1)
print(outputs)

print("predicted:",' '.join(f'{classes[predicted[j]]:5s}' for j in range(4)))

correct = 0
total = 0
# 학습 중이 아니므로, 출력에 대한 변화도를 계산할 필요가 없습니다
with torch.no_grad():
    for data in valid_loader:
        images, labels = data
        # 신경망에 이미지를 통과시켜 출력을 계산합니다
        outputs = net(images)
        # 가장 높은 값(energy)를 갖는 분류(class)를 정답으로 선택하겠습니다
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        #print((predicted))
        #print((labels))
        target = torch.max(labels,1)
        for k in range(4):
           
            #print(target)
            if predicted[k].item() == target.indices[k].item():
                
                correct+=1
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# 변화도는 여전히 필요하지 않습니다
with torch.no_grad():
    for data in valid_loader:
        images, labels = data
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        print("labels: ",labels)
        print("prediction: ",prediction)
        # 각 분류별로 올바른 예측 수를 모읍니다
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# 각 분류별 정확도(accuracy)를 출력합니다
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
print(f'Accuracy of the network on the 18000 test images: {100 * correct // total} %')
'''
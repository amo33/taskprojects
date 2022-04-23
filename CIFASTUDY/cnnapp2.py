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
   
    testimg = testimg.reshape((1, 3, 32, 32))
    testimg = testimg.permute(0,2,3,1).type(torch.float32)

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

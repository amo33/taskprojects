from tkinter import Label
from flask import Flask, jsonify, render_template, request, redirect
import io, os, json 
from PIL import Image
import torch 

app = Flask(__name__)

RESULT_FOLDER = os.path.join('static')
app.config['RESULT_FOLDER'] = RESULT_FOLDER
# model = torch.hub.load('ultralytics/yolov5', 'custom', path='path/to/best.pt')  # 가중치만 로컬, 모델은 본섭 
model = torch.hub.load('./yolov5', 'custom', path='./exp15/weights/best.pt', source='local') # 가중치, 모델 다 로컬
model.eval()
classes = ['car','black_car','white_car']
def prediction(img_bytes):
    img = Image.open(io.BytesIO(img_bytes))
    imgs = [img]
    results = model(imgs, size=640)
    return results 
@app.route("/")
def start():
    return render_template('index.html')

@app.route("/car", methods=['POST'])
def predict():
    pth = "/Users/ihyeonjun/study/yolodetect/runs/detect/exp"
    if os.path.exists(pth):
        os.rmdir(pth)

    error = 'error!'
    
    file = request.files['file']
    if not file:
        return render_template('index.html', data=error)
    img_bytes = file.read()
    results = prediction(img_bytes)
    label = results.pandas().xyxy[0]['class']
    print(label[0])
    print(type(label))
    results.save() #현재 디렉토리에 이미지 저장 
    
    os.rename("/Users/ihyeonjun/study/yolodetect/runs/detect/exp/image0.jpg","/Users/ihyeonjun/study/yolodetect/static/image0.jpg")
    filename = os.path.join(app.config['RESULT_FOLDER'], 'image0.jpg')
  
    return render_template('index.html',result = classes[label[0]], image = filename)

if __name__ == '__main__':
    app.run(debug=True, port=8888)
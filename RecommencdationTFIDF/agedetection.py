"""
This code uses the onnx model to detect faces from live video or cameras.
Use a much faster face detector: https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB
Date: 3/26/2020 by Cunjian Chen (ccunjian@gmail.com)
"""
import time
import cv2
import numpy as np
import onnx
import vision.utils.box_utils_numpy as box_utils
from caffe2.python.onnx import backend

# onnx runtime
import onnxruntime as ort

# import libraries for landmark
from common.utils import BBox, drawLandmark, drawLandmark_multiple
from PIL import Image
import torchvision.transforms as transforms
from utils.align_trans import get_reference_facial_points, warp_and_crop_face

import torch
import torch.nn as nn
#from torch2trt import torch2trt, TRTModule
from torchvision import models, transforms
import glob

import socketio
import json
import os
import onnx
import onnxruntime
import logging
import logging.handlers
import datetime
from pathlib import Path
import insightface
from insightface.app import FaceAnalysis
import pika
from threading import Thread
import paho.mqtt.client as mqtt
from videocaptureasync import VideoCaptureAsync
import atexit
from apscheduler.schedulers.background import BackgroundScheduler


logger = logging.getLogger('TripletFaceEngine')
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    '[%(asctime)s][%(name)s][%(levelname)s] %(message)s')

script_path = os.path.dirname(os.path.realpath(__file__))

log_path = Path(os.path.join(script_path, 'logs'))
if not log_path.exists():
 log_path.mkdir(parents=True)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

file_handler = logging.handlers.TimedRotatingFileHandler(os.path.join(log_path,'default.log'),
                                                            when='midnight',
                                                            interval=1,
                                                            backupCount=30,
                                                            encoding='utf-8')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

client = None


def mq_start():
    logger.info('mq start~~~~~~~~~~~~~~~~~~~~')
    global client
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_message
    client.connect(os.environ['MQ_HOST'], int(os.environ['MQ_PORT']),keepalive=60)
    client.loop_start()


def on_connect(client, userdata, flags, rc):
    logger.info(f'connect mq rc={rc}')
    client.subscribe('face_inquiry', 0)


def on_disconnect(client, userdata, rc):
    logger.info('????????????????????????????????????')
    client.loop_stop()
    client.disconnect()
    mq_start()


def on_message(client, userdata, msg):
    if msg.topic == 'face_inquiry':
        get_mot_bndbox(json.loads(msg.payload))


def ping():
    logger.info(f'$$$$$$$$$$$$$$$$$$$$$ ping $$$$$$$$$$$$')
    client.publish('ping', '')


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# setup the parameters
resize = transforms.Resize([56, 56])
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


crop_size = 112
scale = crop_size / 112.
reference = get_reference_facial_points(default_square=True) * scale
# import the landmark detection models

onnx_model_landmark = onnx.load("onnx/landmark_detection_56_se_external.onnx")
onnx.checker.check_model(onnx_model_landmark)
ort_session_landmark = onnxruntime.InferenceSession(
    "onnx/landmark_detection_56_se_external.onnx", providers=["CUDAExecutionProvider"])


age_gen_classes_org = ['0-9|F', '0-9|M', '10-19|F', '10-19|M', '20-29|F', '20-29|M',
                       '30-39|F', '30-39|M', '40-49|F', '40-49|M', '50-59|F', '50-59|M', '60-69|F', '60-69|M']

rsize = transforms.Resize([224, 224])


model_ft = models.resnet50(pretrained=False)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, len(age_gen_classes_org)))
model_ft.load_state_dict(torch.load
(
    'age_gender_mask_0.85.pth', map_location=device))

model_ft = model_ft.to(device)
model_ft.eval()

model_emp = models.mobilenet_v3_large(pretrained=False)
emp_num_ftrs = model_emp.classifier[-1].in_features
model_emp.classifier[-1] = nn.Linear(emp_num_ftrs, 2)
model_emp.load_state_dict(torch.load('employee-last.pth'), strict=False)
model_emp = model_emp.to(device)
model_emp.eval()


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()




def predict(width, height, confidences, boxes, prob_threshold, iou_threshold=0.3, top_k=-1):
    boxes = boxes[0]
    confidences = confidences[0]
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate(
            [subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = box_utils.hard_nms(box_probs,
                                       iou_threshold=iou_threshold,
                                       top_k=top_k,
                                       )
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]


label_path = "models/voc-model-labels.txt"

onnx_path = "models/onnx/version-RFB-320.onnx"
class_names = [name.strip() for name in open(label_path).readlines()]

predictor = onnx.load(onnx_path)
onnx.checker.check_model(predictor)
onnx.helper.printable_graph(predictor.graph)


predictor = backend.prepare(predictor, device="CUDA:0")
ort_session = ort.InferenceSession(onnx_path)
input_name = ort_session.get_inputs()[0].name


# mot_bndbox -> json data
# { 'id':?, 'bndbox':[x1, y1, x2, y2] }
mot_data = None


def get_mot_bndbox(data):
    global mot_data
    mot_data = data
    logger.info(f'received mot data {mot_data}')
        


mq_start()

ping_scheduler = BackgroundScheduler()
ping_scheduler.start()
ping_scheduler.add_job(func=ping, trigger="interval", seconds=10)

# # Shut down the scheduler when exiting the app
atexit.register(lambda: ping_scheduler.shutdown())


def get_best_age_index(age_list, criterion, count, check=False):
    temp = 0
    if not check:
        for v in age_list:
            temp += v[0] * (1 - abs(criterion - v[0])/100) * v[1]

        best_age_index = temp / count
    else:
        for v in age_list[2:]:
            temp += v[0] * (1 - abs(criterion - v[0])/100) * v[1]
        temp += criterion * age_list[0][1]

        best_age_index = temp / (count - 2)

    return best_age_index


def softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x


def terminate():
    os._exit(1)

def is_employee(img, face_box):
    img_h, img_w, _ = img.shape
    fw = face_box[2]-face_box[0]
    fh = face_box[3]-face_box[1]
    body_x1 = max(face_box[0]-int(fw*1.5), 0)
    body_y1 = face_box[3]
    body_x2 = min(face_box[2]+int(fw*1.5), img_w)
    body_y2 = min(face_box[3]+(fh*4), img_h)
    body = temp_image[body_y1:body_y2, body_x1:body_x2]
    temp_body = body.copy()
    temp_h, _, _ = temp_body.shape
    if temp_h < 200:
        logger.info(f'body height is {temp_h}')
        return 0
    body = Image.fromarray(body)
    body = rsize(body)
    body = to_tensor(body)
    body = normalize(body)
    body.unsqueeze_(0)
    body = body.to(device)
    emp_output = model_emp(body)
    _, emp_predicted = torch.max(emp_output.data, 1)
    emp_predicted = emp_predicted.cpu().numpy()[0]
    logger.info(f'emp predict result {emp_predicted}')
    return int(emp_predicted)

if __name__ == '__main__':

    # cap = cv2.VideoCapture('/home/record/2021-10-20/18_44_20.mp4')  # capture from camera
    # capture from camera
    cap = VideoCaptureAsync(
        'rtsp://admin:triplet1004@192.168.0.8/cam/realmonitor?channel=1&subtype=0')

    threshold = 0.7
    warmup_cnt = 0
    skip_limit = 0.1
    last = time.time()

    detecting_results = {}
    app = FaceAnalysis()
    app.prepare(ctx_id=0)

    FISRT_TIME = True
    cap.start()
    while True:
        try:
            ret, orig_image = cap.read()
            if not ret:
                logger.info("no img")
                terminate()
            # if time.time() - last < skip_limit:
                # continue
            orig_image = orig_image[241:605, 441:718]
            orig_image = cv2.resize(orig_image, None, fx=2, fy=2)
            frame = cv2.cvtColor(orig_image.copy(), cv2.COLOR_BGR2RGB)
            faces = app.get(frame)

            time_time = time.time()

            temp_image = orig_image.copy()

            mapping_mot_box = None  # [id, box]
            
            if mot_data is None or faces is None or len(faces) == 0:
                # cv2.imshow('face', orig_image)
                # cv2.waitKey(1)
                continue
            
            logger.info(
                f'{len(faces)} face(s) detect and {len(mot_data)} mot data(s)')
            box_w = []
            box_x = []
            # logger.info('%%%%%%%%%%%%%%%%%%')
            for face in faces:
                box = face.bbox.astype(np.int)
                # logger.info(box)
                box_w.append([abs(box[2] - box[0]), box])   # [width, box]
                box_x.append([box[0], box])                 # [x1, box]
            box_w = sorted(box_w, key=lambda x: x[0], reverse=True)
            box_x = sorted(box_x, key=lambda x: x[0])

            logger.info('mot_data:{}'.format(mot_data))

            mot_y = []
            mot_x = []

            for data in mot_data:
                mot_y.append([data['id'], data['bndbox'][3]])   # [id, y2]
                mot_x.append([data['id'], data['bndbox'][0]])   # [id, x1]
            mot_y = sorted(mot_y, key=lambda x: x[1], reverse=True)
            mot_x = sorted(mot_x, key=lambda x: x[1])

            # logger.info('box_w: {}'.format(box_w))
            # logger.info('box_x: {}'.format(box_x))
            # logger.info('mot_y: {}'.format(mot_y))
            # logger.info('mot_x: {}'.format(mot_x))

            compare_way = 1     # 1: use width, 2: use x1
            temp_min_y = 10
            if len(mot_data) == len(faces):
                if 1 < len(mot_data):
                    if abs(mot_y[0][1] - mot_y[1][1]) < temp_min_y:
                        compare_way = 2

                if compare_way == 1:
                    logger.info('> 1')
                    mapping_mot_box = [[mot[0], box[1]]
                                       for mot, box in zip(mot_y, box_w)]
                else:
                    logger.info('> 2')
                    mapping_mot_box = [[mot[0], box[1]]
                                       for mot, box in zip(mot_x, box_x)]
            elif len(faces) < len(mot_data):
                if 1 < len(mot_data):
                    if abs(mot_y[0][1] - mot_y[1][1]) < temp_min_y:
                        compare_way = 2

                if compare_way == 1:
                    logger.info('> 3')
                    mapping_mot_box = [[mot_y[0][0], box_w[0][1]]]
                else:
                    logger.info('> 4')
                    mapping_mot_box = [[mot[0], box[1]]
                                       for mot, box in zip(mot_x[:len(box_x)], box_x)]
            elif len(mot_data) < len(faces):
                logger.info('> 5')
                mot_img_x_size = 800
                x_size = 480

                mot_x = [[mot[0], mot[1]/mot_img_x_size]
                         for mot, mot in zip(mot_x, mot_x)]
                box_x = [[box[0]/x_size, box[1]]
                         for box, box in zip(box_x, box_x)]

                mapping_mot_box = []
                for mot in mot_x:
                    compare_x = [[abs(mot[1]-box[0]), box[1]] for box in box_x]
                    compare_x = sorted(compare_x, key=lambda x: x[0])

                    mapping_mot_box.append([mot[0], compare_x[0][1]])

            logger.info('mapping_mot_box: {}'.format(mapping_mot_box))

            predict_result = []

            logger.info('detecting success')
            age_gen_start = time.time()
            for value in mapping_mot_box:
                box = value[1]
                # cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)

                # perform landmark detection
                # out_size = 56
                # img=orig_image.copy()
                # height,width,_=img.shape
                # x1=box[0]
                # y1=box[1]
                # x2=box[2]
                # y2=box[3]
                # w = x2 - x1 + 1
                # h = y2 - y1 + 1
                # size = int(max([w, h])*1.1)
                # cx = x1 + w//2
                # cy = y1 + h//2
                # x1 = cx - size//2
                # x2 = x1 + size
                # y1 = cy - size//2
                # y2 = y1 + size
                # dx = max(0, -x1)
                # dy = max(0, -y1)
                # x1 = max(0, x1)
                # y1 = max(0, y1)

                # edx = max(0, x2 - width)
                # edy = max(0, y2 - height)
                # x2 = min(width, x2)
                # y2 = min(height, y2)
                # new_bbox = list(map(int, [x1, x2, y1, y2]))
                # new_bbox = BBox(new_bbox)
                # cropped=img[new_bbox.top:new_bbox.bottom,new_bbox.left:new_bbox.right]
                # if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
                #     cropped = cv2.copyMakeBorder(cropped, int(dy), int(edy), int(dx), int(edx), cv2.BORDER_CONSTANT, 0)
                # cropped_face = cv2.resize(cropped, (out_size, out_size))

                # if cropped_face.shape[0]<=0 or cropped_face.shape[1]<=0:
                #     cv2.destroyAllWindows()
                #     continue
                # cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
                # cropped_face = Image.fromarray(cropped_face)
                # test_face = resize(cropped_face)
                # test_face = to_tensor(test_face)
                # test_face = normalize(test_face)
                # test_face.unsqueeze_(0)

                # start = time.time()
                # ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(test_face)}
                # ort_outs = ort_session_landmark.run(None, ort_inputs)
                # end = time.time()

                # landmark = ort_outs[0]
                # landmark = landmark.reshape(-1,2)
                # landmark = new_bbox.reprojectLandmark(landmark)

                # # crop and aligned the face
                # lefteye_x=0
                # lefteye_y=0
                # for i in range(36,42):
                #     lefteye_x+=landmark[i][0]
                #     lefteye_y+=landmark[i][1]
                # lefteye_x=lefteye_x/6
                # lefteye_y=lefteye_y/6
                # lefteye=[lefteye_x,lefteye_y]

                # righteye_x=0
                # righteye_y=0
                # for i in range(42,48):
                #     righteye_x+=landmark[i][0]
                #     righteye_y+=landmark[i][1]
                # righteye_x=righteye_x/6
                # righteye_y=righteye_y/6
                # righteye=[righteye_x,righteye_y]

                # # 얼굴 정면이 아닐 경우 걸러내버림
                # # if 100 <= abs(lefteye[0] - lefteye[1]): continue
                # # if 100 <= abs(righteye[0] - righteye[1]): continue

                # nose=landmark[33]
                # leftmouth=landmark[48]
                # rightmouth=landmark[54]
                # facial5points=[righteye,lefteye,nose,rightmouth,leftmouth]
                # # warped_face = warp_and_crop_face(np.array(temp_image), facial5points, reference, crop_size=(crop_size, crop_size))

                warped_face = cv2.resize(

temp_image[max(box[1]-10, 0):min(box[3]+10, temp_image.shape[0]), max(
                    box[0]-10, 0):min(box[2]+10, temp_image.shape[1])], (crop_size, crop_size))
                save_face = warped_face.copy()
                warped_face = Image.fromarray(warped_face)
                warped_face = rsize(warped_face)
                warped_face = to_tensor(warped_face)
                warped_face = normalize(warped_face)
                warped_face.unsqueeze_(0)
                warped_face = warped_face.to(device)
                # age_gen_start = time.time()
                age_gen_outputs = model_ft(warped_face)
                _, age_gen_predicted = torch.max(age_gen_outputs.data, 1)
                # print(age_gen_outputs.cpu().detach().numpy().squeeze())

                temp_out = age_gen_outputs.cpu().detach().numpy().squeeze()
                label_val = age_gen_classes_org[age_gen_predicted.item()].replace(
                    '|', '_')
                logger.info('####################### {}_{}'.format(value[0], label_val))
                # now = datetime.datetime.now()
                # p = Path('results/{}'.format(now.strftime('%Y-%m-%d')))
                # p.mkdir(exist_ok=True, parents=True)
                # filename = '{}_{}_{}.jpg'.format(int(time.time()*1000),value[0],label_val)
                # cv2.imwrite('{}/{}'.format(str(p), filename), save_face)

                temp_out = softmax(temp_out)
                sort_temp_out = sorted(temp_out, reverse=True)

                arg_new_temp_out = (-temp_out).argsort()    # 확률 큰 순으로 index 정렬
                test = (-temp_out).argsort()
                test = [age_gen_classes_org[v] for v in test]

                # gender = 'M'
                # male = 0
                # female = 0
                # for result in test[:3]:
                #     index_gender = result.split('|')[-1]
                #     if index_gender == 'M': male += 1
                #     elif index_gender == 'F': female += 1

                # logger.info('m:{} vs f:{}'.format(male,female))
                # if male < female: gender = 'F'
                gender = test[0].split('|')[-1]

                age = int(test[0].split('|')[0].split('-')[0])

                #predict_result_txt = test[0] + '|' + gender
                #cv2.putText(orig_image, predict_result_txt, (box[0], box[1]+30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 1, cv2.LINE_AA)

                # logger.info('detecting {} >> {}'.format(
                    # len(mapping_mot_box), time.time()-age_gen_start))

                if value[0] not in detecting_results:
                    detecting_results[value[0]] = {}

                if 'age' not in detecting_results[value[0]]:
                    detecting_results[value[0]]['age'] = {}
                if 'gender' not in detecting_results[value[0]]:
                    detecting_results[value[0]]['gender'] = {}

                #result = age_range + '|' + gender
                #print('result:', result)

                # if result not in detecting_results[value[0]]:
                #    detecting_results[value[0]][result] = 0

                if age not in detecting_results[value[0]]['age']:
                    detecting_results[value[0]]['age'][age] = 0
                if gender not in detecting_results[value[0]]['gender']:
                    detecting_results[value[0]]['gender'][gender] = 0

                detecting_results[value[0]]['age'][age] += 1
                detecting_results[value[0]]['gender'][gender] += 1

                #best = sorted(detecting_results[value[0]].items(), key=lambda x: x[1], reverse=True)
                #best_result = max(detecting_results[value[0]].keys(), key=lambda x: detecting_results[value[0]][x])

                logger.info(detecting_results[value[0]]['gender'])
                best_age_index = max(detecting_results[value[0]]['age'].keys(
                ), key=lambda x: detecting_results[value[0]]['age'][x])
                age_list = sorted(
                    detecting_results[value[0]]['age'].items(), key=lambda x: x[1], reverse=True)
                best_gen_index = max(detecting_results[value[0]]['gender'].keys(
                ), key=lambda x: detecting_results[value[0]]['gender'][x])

                logger.info('detecting_age_results: {}'.format(age_list))
                logger.info('>>>>>>> best_age: {}'.format(best_age_index))

                count = sum(detecting_results[value[0]]['age'].values())
                emp = is_employee(temp_image.copy(), box)
                

                if 1 == count:
                    logger.info('>>>>> Here is 1: {}'.format(best_age_index))

                    predict_result.append(
                        {'id': value[0], 'gender': best_gen_index, 'age': best_age_index, 'emp': emp})
                    try:
                        client.publish(
                            'face_result', payload=json.dumps(predict_result))
                    except:
                        mq_start()
                        client.publish(
                            'face_result', payload=json.dumps(predict_result))
                elif 3 == count:
                    criterion = age_list[0][0]
                    if len(age_list) == 3:  # 1 1 1
                        age_list = sorted(
                            detecting_results[value[0]]['age'].items(), key=lambda x: x[0])
                        criterion = age_list[1][0]  # 중앙값

                    best_age_index = get_best_age_index(
                        age_list, criterion, count)
                    logger.info('>>>>> Here is 3: {}'.format(best_age_index))

                    predict_result.append(
                        {'id': value[0], 'gender': best_gen_index, 'age': best_age_index, 'emp': emp})
                    try:
                        client.publish(
                            'face_result', payload=json.dumps(predict_result))
                    except:
                        mq_start()
                        client.publish(
                            'face_result', payload=json.dumps(predict_result))
                elif 5 == count:
                    criterion = age_list[0][0]   # 기본: 가장 많이 나온 값
                    check = False   # 2 1 1 check
                    if len(age_list) == 5:  # 1 1 1 1 1
                        age_list = sorted(
                            detecting_results[value[0]]['age'].items(), key=lambda x: x[0])
                        criterion = age_list[2][0]   # 중앙값 뽑기
                    # 2 2 1
                    elif len(age_list) == 3 and age_list[0][1] == age_list[1][1]:
                        criterion = round(
                            (age_list[0][0] + age_list[1][0]) / 2)
                        check = True

                    best_age_index = get_best_age_index(
                        age_list, criterion, count, check)

                    logger.info('>>>>> Here is 5: {}'.format(best_age_index))

                    #age, gender = best_result.split('|')
                    predict_result.append(
                        {'id': value[0], 'gender': best_gen_index, 'age': best_age_index, 'emp': emp})

                    logger.info('predict: {}'.format(predict_result))
                    try:
                        client.publish(
                            'face_result', payload=json.dumps(predict_result))
                    except:
                        mq_start()
                        client.publish(
                            'face_result', payload=json.dumps(predict_result))

                    detecting_results[value[0]] = {}
                else:
                    if count > 0:
                        logger.info('>>>>> count {}: {}'.format(
                            count, best_age_index))
                        predict_result.append(
                            {'id': value[0], 'gender': best_gen_index, 'age': best_age_index, 'emp': emp})
                        try:
                            client.publish(
                                'face_result', payload=json.dumps(predict_result))
                        except:

                            mq_start()
                            client.publish(
                                'face_result', payload=json.dumps(predict_result))
                            logger.info('------------')

                            mot_data = None

                            # if warmup_cnt > 1:
                            # orig_image = cv2.resize(orig_image, (0, 0), fx=0.7, fy=0.7)
                            # cv2.imshow('face', orig_image)
                            # cv2.waitKey(1)
                            # if cv2.waitKey(1) & 0xFF == ord('q'):
                            # break
                            # else:
                            # warmup_cnt += 1
                            last = time.time()
        except Exception as e:
            print(e)
            terminate()
    cap.stop()
    cv2.destroyAllWindows()
    terminate()
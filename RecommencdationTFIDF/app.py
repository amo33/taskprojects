from builtins import print
from unicodedata import category
from flask import Flask, render_template, request, redirect, url_for, jsonify, make_response
import detectron2
from detectron2.utils.logger import setup_logger
import os
import numpy as np
import cv2
import random
import torch
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode
from detectron2.data.datasets import register_coco_instances
from detectron2.structures import Instances
from detectron2.utils.visualizer import Visualizer, VisImage
from PIL import Image
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data.transforms import Transform
import matplotlib.pyplot as plt
from pathlib import Path
import docker
import re
import time
import datetime
import pyspark 
from pyspark.sql.functions import monotonically_increasing_id 
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import concat, col, lit
import pyspark.sql.functions as F
from pyspark.sql.functions import split, col
from pyspark.ml.feature import HashingTF, IDF
from pyspark.ml.feature import Normalizer
from pyspark.sql.types import StructType,StructField,ArrayType, StringType, LongType
import io
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib
from math import dist
import json 
from pyspark.sql.types import DoubleType
import uuid
import glob
import base64
from pytz import timezone
import redis
import sys
from flask_func import *
import paho.mqtt.client as mqtt
import logging
from logging.handlers import TimedRotatingFileHandler
import atexit 

from apscheduler.schedulers.background import BackgroundScheduler
from pyspark.sql.functions import rand


df_woman = spark.read.json("../deepfashion/dataset/Femaledata/female_style.json")
df_man = spark.read.json("../deepfashion/dataset/Maledata/male_style.json")
df_woman = df_woman.dropDuplicates()
df_man = df_man.dropDuplicates()
startTime = time.time()

def shuffle_dataframe():
	global df_man
	global df_woman 
	df_man = df_man.orderBy(rand()) # shuffle 
	df_woman = df_woman.orderBy(rand()) 
# scheduler = BackgroundScheduler()
# scheduler.add_job(shuffle_dataframe, 'cron', minute='*/5', second='10, 30') # daemon으로 호출 
# scheduler.start()
top_category = ['니트웨어','아우터','상의','셔츠','블라우스','베스트','점퍼','티셔츠','드레스','브라탑','후드티','가디건']
bottom_category = ['팬츠','하의','스커트','청바지','레깅스','조거팬츠',"trousers"] 

spark = SparkSession\
		.builder\
		.appName('Python Spark basic example')\
		.config('spark.some.config.option', 'some-value')\
		.getOrCreate()


spark.conf.set("spark.sql.autoBroadcastJoinThreshold", -1)
spark.conf.set("spark.python.worker.memory", '24g')

result_2 =[(None,'트래디셔널',int(df_man.collect()[-1][3])+1, '슬릿',"저지",'트래디셔널,캐주얼,컨템포러리', "긴바지")]
extract_top_lst(result_2,df_woman, 'Femaledata')
extract_bottom_lst(result_2, df_man, 'Maledata')
extract_top_lst(result_2, df_man, 'Maledata')
extract_bottom_lst(result_2, df_woman, 'Femaledata')
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
app.config['MAX_CONTENT_LENGTH'] = 32 * 1000 * 1000
app.config['DEBUG'] = False
r = redis.Redis(host=os.environ['REDIS_IP'], port=os.environ['REDIS_PORT'], decode_responses=True)
os.environ["CUDA_VISIBLE_DEVICES"]="1"
script_path = os.path.dirname(os.path.realpath(__file__))
logger = logging.getLogger('DeepLoungeSServer')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s [%(levelname)s][%(filename)s:%(lineno)s - %(funcName)20s() ] %(name)s: %(message)s')
log_path = Path(os.path.join(script_path, 'logs'))
if not log_path.exists():
	log_path.mkdir(parents=True)

handler = TimedRotatingFileHandler(
	os.path.join(script_path, 'logs', 'default.log'),
	when='midnight',
	interval=1,
	encoding='utf-8'
)
handler.suffix = '%Y-%m-%d'
app.config['DEBUG'] = False

handler.setFormatter(formatter)
logger.addHandler(handler)

print ('++++++++++++++++++++++++++++++')
print(r.ping())
# r.set('flag', 'warmup')
# print(r.get('flag'))
cfg = get_cfg()

cfg.MODEL.DEVICE = "cuda"
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = "../deepfashion/output/model_50_0523.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 13
register_coco_instances("deepfashion_train", {}, "../deepfashion/dataset7000/train/deepfashion2_7000.json", "../deepfashion/dataset7000/train/image")
cfg.DATASETS.TEST = ("deepfashion_train", )
img = cv2.imread("/home/deep_lounge_S/static/Imagedata/63.jpg")
predictor = DefaultPredictor(cfg)
outputs = predictor(img)




# deptSchema = StructType([       
#     StructField('스타일', StringType(), True),
#     StructField('카테고리', StringType(), True),
#     StructField('소재', StringType(), True),
#     StructField('file_path', StringType(), True),
#     StructField('id', LongType(), True)
# ])
# deptSchema = StructType([
#     StructField('file_path', StringsType(), True),
#     StructField('file_style_path', StringType(), True), 
#     StructField('id', LongType(), True), 
#     StructField('디테일', StringType(), True),
#     StructField('소재', StringType(), True),
#     StructField('스타일', StringType(), True),
#     StructField('카테고리', StringType(), True), 
# ])



client = None
def mq_start():
	global client
	client = mqtt.Client()
	client.on_connect = mq_on_connect
	client.on_disconnect = mq_on_disconnect
	client.on_message = mq_on_message
	client.connect(os.environ['MQ_HOST'], int(os.environ['MQ_PORT']))
	client.loop_start()


def mq_on_connect(client, userdata, flags, rc):
	logger.info(f'connect mq rc={rc}')
	client.subscribe('still',0)
	client.subscribe('heartbeat',0)
	client.subscribe('send_me_res',0)
	

def mq_on_disconnect(client, userdata, rc):
	logger.info(f'disconnected mq rc={rc}')
	mq_start()

def mq_on_message(client, userdata, msg):
	logger.info(f'======{msg.topic}')




@app.route('/')
def main():
	return render_template('main.html')

# test
@app.route('/share2', methods=['POST'])
def send_mail2():
	print(request)
	print(request.get_json())
	print(type(request.get_json()))

	return jsonify({'success':'true'})

# ali : 공유하기_메일 전송 기능
@app.route('/share', methods=['POST'])
def send_mail():
	logger.info('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ mail')
	logger.info('!@!@!@!@EWFQFQE')
	logger.info(request)
	logger.info(request.get_json())

	request_dict = request.get_json()
	dest = request_dict['email']
	texture = request_dict['texture']
	detail = request_dict['pattern']
	style_list = request_dict['style']
	# style_list = style_list_str.split(', ')
	style = ''
	exec_path = os.path.dirname(os.path.abspath(__file__))

	for s in style_list :
		temp = '<div style="font-size: 3vw; font-family: Pretendard; color: #000">{}</div>\n'.format(s)
		style += temp

	# 받는 사진
	#dest = "wakebro119@naver.com"
	recommendation_path = 'static/recommend'

	# 로그 기록
	now_time = datetime.datetime.now()
	now_time = datetime.datetime.strftime(now_time, '%Y-%m-%d %H:%M:%S.%f') 
	file_name = now_time.split(' ')[0]
	file_path = 'static/sendMaillogs/{}.txt'.format(file_name)

	os.makedirs('static/sendMaillogs', exist_ok=True)
	if not os.path.isfile(file_path) :
		f = open(file_path, 'w')
	else :
		f = open(file_path, 'a')
	f.write('{} : {}\n'.format(now_time, dest))
	f.write('--------------------------------------------------------------------\n')
	f.close()

	data = MIMEMultipart()
	data['Subject'] = '[트리플렛]고객님께서 공유하신 추천코디 정보입니다.'

	# logo 상단
	logo_path = os.path.join(exec_path, 'static', 'img', 'DeepLounge_BI_S.png')
	fp = open(logo_path, 'rb')
	msgImage = MIMEImage(fp.read())
	fp.close()
	msgImage.add_header('Content-ID', '<logo>')
	data.attach(msgImage)

	# logo 하단
	logo2_path = os.path.join(exec_path, 'static', 'img', 'triplet_ci_dark_mini.png')
	fp = open(logo2_path, 'rb')
	msgImage = MIMEImage(fp.read())
	fp.close()
	msgImage.add_header('Content-ID', '<logo2>')
	data.attach(msgImage)
	count = 0

	input_html = ''
	for idx, path in enumerate(os.listdir(recommendation_path)) :
		if path == '.DS_Store' :
			continue
		full_path = os.path.join(recommendation_path, path)
		img_name = '<img' + str(idx + 1) + '>'
		
		
		fp = open(full_path, 'rb')
		msgImage = MIMEImage(fp.read())
		fp.close()
		msgImage.add_header('Content-ID', img_name)
		data.attach(msgImage)
		
		temp = '<img src="cid:{}" width="30%" style="margin: 20px;">\n'.format('img' + str(idx+1))
		input_html += temp
		#print('{} 파일 경로 : {}'.format(img_name, full_path))
		count += 1

	full_html = \
		'''
	<html>
	<head>
		<link rel="stylesheet" as="style" crossorigin href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css" />
	</head>
	<body style="width:100%; margin: 0 auto;">
		<div class="container" style="background-color: #f3f3f3; padding: 30px; ">
			<div style="background-color: #fff;">
				<header style="padding: 5%;">
					<img src="cid:logo" width="40%">
				</header>
				<section>
					<article style="
						width: 80%;
						margin: 20px 0 20px 30px;
						font-family: Pretendard;
						font-size: 3.5vw;
						font-weight: 600;
						font-stretch: normal;
						font-style: normal;
						line-height: normal;
						letter-spacing: normal;
						color: #000;">
						<span style="margin-left: 2%;">코엑스점<span style="font-weight: 300; color: #3f3e43;">에서 발송한</span></span><br/>
						<span style="margin-left: 2%;">AI 추천 스타일링 {}종<span style="font-weight: 300; color: #3f3e43;">입니다.</span></span>
					</article>
					<br/>
					<article style="padding-left: 5%; text-align: left;">
						<div style="width: 20%; display: inline-block; vertical-align: top; margin: 10px">
							<div style="font-size: 2vw; font-family: Pretendard; color: #000;">스타일</div>
							{}
						</div>
						<div style="width: 20%; display: inline-block; vertical-align: top; margin: 10px">
							<div style="font-size: 2vw; font-family: Pretendard; color: #000;">텍스처</div>
							<div style="font-size: 3vw; font-family: Pretendard; color: #000;">{}</div>
						</div>
						<div style="width: 20%; display: inline-block; vertical-align: top; margin: 10px">
							<div style="font-size: 2vw; font-family: Pretendard; color: #000;">디테일</div>
							<div style="font-size: 3vw; font-family: Pretendard; color: #000;">{}</div>
						</div>
					</article>
					<br/>
					<article style="padding: 40px; text-align: center;">
						{}
					</article>
					<br/>
				</section>
			</div>
			<footer style="
				font-family: 'Pretendard';
				font-size: 18px;
				font-weight: normal;
				font-stretch: normal;
				font-style: normal;
				line-height: 1.44;
				letter-spacing: normal;
				text-align: center;
				color: #3f3e43;">
				<br/><br/>
				<span>혹시 문의가 있으세요?</span><br/>
				<div style="margin-top: 20px;">
					<span><strong>help@tiplllet.com</strong>으로</span><br/>
				</div>
				<span>문의주시면 답변 드리겠습니다.</span><br/>
				<span>감사합니다.</span><br/>
				<img src="cid:logo2" width="100px" style="margin-top: 30px;">
			</footer>
		</div>
	</body>
	</html>
		'''.format(count, style, texture, detail, input_html)

	msg = MIMEText(full_html, 'html')
	data.attach(msg)

	session = smtplib.SMTP_SSL('smtp.gmail.com', 465)
	session.login('tank@triplllet.com', '=dUa%3qM')
	session.sendmail('tank@triplllet.com', dest, data.as_string())
	session.quit()

	logger.info('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ mail send success')

	return jsonify({'success':'true'})

@app.route('/detection')
def detection():
	return render_template('detection.html')

@app.route('/sendImage', methods=['POST'])
def sendImage():
	path = "./result"
	file_list = os.listdir(path)
	if os.path.exists("result") : 
		for file in os.scandir("result") : 
			os.remove(file.path)
	startTime = time.time()
	totalTime = time.time()
	global client
	path = "/home/deep_lounge_S/result"
	data = base64.b64decode(request.form['image'])
	image = Image.open(io.BytesIO(data))
	image_np = np.array(image)
	image_np = cv2.rotate(image_np,cv2.ROTATE_90_COUNTERCLOCKWISE)
	logger.info("[DONE] print image_time spent :{:.4f}".format(time.time() - startTime))
	cv2.imwrite('test.jpg',image_np)
	# img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
	model1Time = time.time()
	predictor = DefaultPredictor(cfg)
	outputs = predictor(image_np)
	mask_array = outputs['instances'].pred_masks.detach().cpu().numpy()
	mask_array_temp = np.array(0)
	num_instances = mask_array.shape[0]
	result = {}
	top ={}
	bottom = {}
	
	lst_name = ['short_sleeved_shirt', 'long_sleeved_shirt', 'short_sleeved_outwear', 'long_sleeved_outwear',
				'vest', 'sling', 'shorts', 'trousers', 'skirt', 'short_sleeved_dress',
				'long_sleeved_dress', 'vest_dress', 'sling_dress'] 

	t = ['short_sleeved_shirt', 'long_sleeved_shirt','short_sleeved_outwear','long_sleeved_outwear','vest',
		'sling', 'short_sleeved_dress', 'long_sleeved_dress', 'vest_dress', 'sling_dress']

	b = ['shorts', 'trousers', 'skirt']
	avg = (960, 540)
	if num_instances > 0:
		logger.info(f'===============total mask : {num_instances}')
		for i, data in enumerate(outputs["instances"].pred_classes.detach().cpu().numpy().tolist()) :
			if lst_name[data] in t : 
				logger.info(f'===============index : {i} category : 상의')
			if lst_name[data] in b : 
				logger.info(f'===============index : {i} category : 하의')
			im = Image.fromarray(img)
			x_min , y_min, x_max, y_max = outputs["instances"].pred_boxes.tensor[i].detach().cpu().numpy()
			x_min = int(x_min)
			x_max = int(x_max)
			y_max = int(y_max)
			y_min = int(y_min)
			coordinate = (int((x_min+x_max)/2), int((y_min+y_max)/2))
			def go_save() : 
				if lst_name[data] in t : 
					top['category'] = '상의'
					top['coordinate'] = str(coordinate)
					category = 'top'
				else : 
					bottom['category'] = '하의'
					bottom['coordinate'] = str(coordinate)
					category = 'bottom'
				mask_array_temp = np.array(0)
				crop_img = im.crop((x_min, y_min, x_max, y_max))
				mask_array_temp = mask_array[i][y_min:y_max, x_min:x_max]
				mask_array_temp = mask_array_temp.reshape((1,) + mask_array_temp.shape)
				mask_array_temp = np.moveaxis(mask_array_temp, 0, -1)
				mask_array_temp = np.repeat(mask_array_temp, 3, axis=2)
				output = np.where(mask_array_temp==False, 255, crop_img)
				img_process = output
				img_process = Image.fromarray(img_process)
				img_process = img_process.convert("RGBA") # 뒷배경 투명을 위해 RGBA로 엽니다. 
				datas = img_process.getdata()
				newData = []
				for item in datas:
					if item[0] > 255 and item[1] > 255 and item[2] > 255:
						newData.append((item[0], item[1], item[2], 0)) # 뒤에꺼는 0으로 넣어줍니다. (결과는 포토샵처럼 나옵니다.)
					else:
						newData.append(item)

				img_process.putdata(newData)
				img_process.save("result/masked_"+category +".png") 


				logger.info(f'lst_name={lst_name[data]}')
			if lst_name[data] in t :
				logger.info('top')
				if top.get('category') == None:
					logger.info('top is none, save')
					go_save()
				else:
					old_data_x = int(top.get('coordinate').strip("("")").split(",")[0])
					old_dist = abs(avg[0]-old_data_x)
					new_dist = abs(avg[0]-coordinate[0])
					logger.info(f'old dist={old_dist}, new_dist={new_dist}')
					if old_dist >  new_dist: 
						logger.info('new closer item saved')
						go_save()
					else:
						logger.info('old is more closer')
			else:
				logger.info('bottom')
				if bottom.get('category') == None:
					if coordinate[1] >600 :
						logger.info('bottom is none and coord over 600')
						go_save()
					else : 
						logger.info(f'pass : {coordinate[1]}')
				else:
					if coordinate[1] >600 : 
						old_data_x = int(bottom.get('coordinate').strip("("")").split(",")[0])
						old_dist = abs(avg[0]-old_data_x)
						new_dist = abs(avg[0]-coordinate[0])
						logger.info(f'old dist={old_dist}, new_dist={new_dist}')
						if old_dist >  new_dist: 
							logger.info('new closer item saved')
							go_save()
						else:
							logger.info('old is more closer')
					else : 
						logger.info(f'pass : {coordinate[1]}')
							

			# if top.get('category') != None or bottom.get('category') != None : 
			# 	if top.get('category') != None : 
			# 		if lst_name[data] in t : 
			# 			old_data_x = int(top.get('coordinate').strip("("")").split(",")[0])
			# 			old_dist = abs(avg[0]-old_data_x)
			# 			new_dist = abs(avg[0]-coordinate[0])
			# 			# if dist(avg, old_data) > dist(avg, chk) :
			# 			logger.info(f'old dist={old_dist}, new_dist={new_dist}')
			# 			if abs(avg[0]-old_data_x) > abs(avg[0]-coordinate[0]) : 
			# 				go_save()	
			# 		else : 
			# 			if coordinate[1] >600 : 
			# 				go_save()
			# 	elif bottom.get('category') != None : 
			# 		if lst_name[data] in b : 
			# 			old_data_x = int(bottom.get('coordinate').strip("("")").split(",")[0])
			# 			old_dist = abs(avg[0]-old_data_x)
			# 			new_dist = abs(avg[0]-coordinate[0])
			# 			logger.info(f'old dist={old_dist}, new_dist={new_dist}')
			# 			if old_dist >  new_dist: 
			# 				if coordinate[1] >600 : 
			# 					go_save()
			# 				else:
			# 					logger.info(f'new bottom coordinate-{coordinate[1]}')
			# 		else : 
			# 			go_save()
			# else : 
			# 	if lst_name[data] in b : 
			# 		if coordinate[1] >600 : 
			# 			go_save()
			# 		else : 
			# 			logger.info(f'new bottom coordinate-{coordinate[1]}')
			# 	else : 
			# 		go_save()
		logger.info("[DONE] print model1_time spent :{:.4f}".format(time.time() - model1Time))
		model2Time = time.time()
		path = './result'
		file_list = os.listdir(path)
		logger.info(f'==========file_list :  {file_list}')
		for file_name in file_list : 
			if 'top' in file_name :
				client.publish('top','')
				logger.info(f'category =  top')
			if 'bottom' in file_name :
				client.publish('bottom','')
				logger.info(f'category =  bottom')
			r.set('style', '')
			r.set('detail', '')
			r.set('texture', '')
			s_data = ''
			d_data = ''
			t_data = ''

			temp_dict = {}
			while True:
				s_data = r.get('style')
				if s_data != '': temp_dict['style'] = s_data
				t_data = r.get('texture')
				if t_data != '': temp_dict['texture'] = t_data
				d_data = r.get("detail")			
				if d_data != '': temp_dict['detail'] = d_data
				if len(temp_dict.keys()) == 3 and temp_dict['style'] != '' and temp_dict['detail'] != '' and temp_dict['texture'] != '': break 

			if 'top' in file_name : 
				top['style'] = s_data
				top['detail'] = d_data
				top['texture'] = t_data
			if 'bottom' in file_name : 
				bottom['style'] = s_data
				bottom['detail'] = d_data
				bottom['texture'] = t_data
		logger.info("[DONE] print model2_time spent :{:.4f}".format(time.time() - model2Time))
	
	result['top'] = top
	result['bottom'] = bottom
	if 'category' not in result['top']:
		result['top']['category'] = ''
		result['top']['coordinate'] = '(0,0)'
		result['top']['style'] =''
		result['top']['detail'] = ''
		result['top']['texture'] = ''
	if 'category' not in result['bottom']:
		result['bottom']['category'] = ''
		result['bottom']['style'] =''
		result['bottom']['texture'] =''
		result['bottom']['detail'] =''
		result['bottom']['coordinate'] = '(0,0)'

	logger.info(result)
	logger.info("[DONE] print sendImage_total spent :{:.4f}".format(time.time() - totalTime))
	return make_response(jsonify(result))	

	#request: base64 image
	# print(request.get_json()['image'])
 
	#return: Analysis
	# return make_response(jsonify({'top':{'category':'상의','coordinate':'(34,23)','detail':'포켓','texture':'우븐','style':'TRADITIONAL, MANISH, FEMININE' },
	# 		'bottom':{'category':'하의','coordinate':'(34,23)','detail':'포켓','texture':'우븐','style':'TRADITIONAL, MANISH, FEMININE' } }))

@app.route('/sendResult', methods=['POST'])
def sendResult(): # 추천 코디를 재조합해주는 api 
	
	global df_woman # 여성 스파크 데이터와 남성 스파크 데이터는 모두 전역 변수로 불러왔습니다. df_woman, df_man 
	global df_man
	
	params = request.form.to_dict() # 프론트로부터 정보를 받아옵니다. 
	startTime = time.time() # shuffle하는 시간 측정 
	shuffle_dataframe() # 이 코드는 남성과 여성 데이터를 모두 shuffle해주는 코드입니다. 해당 함수는 제가 커스텀으로 만든 함수입니다. 
	logger.info("shuffle time")
	logger.info(time.time()-startTime) # shuffle 시간도 측정 

	# 아래 2개의 주석은 프론트로부터 받아오는 데이터의 예시로 테스트할때 사용했습니다. 
	#param = {'gender': '{gender=F}', 'bottom': '{style=GENDERLESS,CASUAL, detail=스트링, category=팬츠, texture=우븐}', 'top': '{style=매니쉬,캐주얼, detail=드롭숄더, category=셔츠, texture=니트}'}
	#{'gender': '{gender=여성}', 'bottom': '{style=팬츠, detail=팬츠, category=팬츠, texture=팬츠}', 'top': '{style=팬츠, detail=팬츠, category=팬츠, texture=팬츠}'}

	top_category = ['니트웨어','아우터','상의','셔츠','블라우스','베스트','점퍼','티셔츠','드레스','브라탑','후드티','가디건'] # 상의 카테고리 종류
	bottom_category = ['팬츠','하의','스커트','청바지','레깅스','조거팬츠',"긴바지"] # 하의 카테고리 종류 

	logger.info(params) # 값 받아온 형태 로그로 출력 
	logger.info(params['gender'].split('=')[1][:-1]) # 성별 추출 : 여성 or 남성 
	gender = 'Femaledata' if params['gender'].split('=')[1][:-1] == '여성' else 'Maledata' # 남성이면 maledata, 여성이면 Femaledata로 gender를 정의합니다. 
	logger.info(gender) # 성별 체크 

	result_bottom = [] # 하의 정보들
	result_top = [] # 상의 정보들 
	# 아래 두 조건문은 프론트에서 들어온다면 거기서 replace로 데이터 파싱해서 result_bottom, result_top에 넣어준다.
	if 'bottom' in params.keys():
		datum = []
		dilim = params['bottom'].replace("style=","/").replace(", detail=","/").replace(", category=","/").replace(", texture=","/").replace("}","/")
		parsed_data = dilim.split("/")

		datum.append(None)
		datum.append('temp')
		datum.append(-1)
		datum.append(parsed_data[2])
		datum.append(parsed_data[4])
		datum.append(convertstyle(parsed_data[1],'eng')) # style 
		datum.append(parsed_data[3])

		datum = tuple(datum)
		result_bottom.append(datum)
	
	if 'top' in params.keys():
		datum = []
		dilim = params['top'].replace("style=","/").replace(", detail=","/").replace(", category=","/").replace(", texture=","/").replace("}","/")
		datum.append(None)
		parsed_data = dilim.split("/")
		datum.append('temp')
		datum.append(-1)
		datum.append(parsed_data[2])
		datum.append(parsed_data[4])
		datum.append(convertstyle(parsed_data[1],'eng'))
		datum.append(parsed_data[3])
		datum = tuple(datum)
		result_top.append(datum)
	#result_top = [(None,'temp',-1, '슬릿',"저지",'트래디셔널,서브컬쳐,캐주얼', "셔츠")]와 비슷하게 데이터들이 저장되는지 확인해보면 좋을거 같습니다. 한가지 말씀 못 드럈던 점은 datum이 tuple형태로 들어가서 [()] 안에 정보들이 있습니다. 
	# 값이 중간에 변경되는 경우를 막기 위해 tuple형태로 사용하게 되었습니다.

	# top(상의)에 대한 정보들을 각각 top_list, top_info에 넣어준다. 
	# 여기서 top_)info는 category, style, textrue, detail을 포함한 모두 정보를 갖고 있고, top_list는 file의 고유명과 고유폴더명을 저장하고 있다. 
	# bottom도 마찬가지로 만들어주는 과정입니다. 

	if gender == "Maledata":
		if result_top[0][4] != '':
			top_list, top_info =extract_top_lst(result_top,df_man, gender)
		else:
			top_list= []
			top_info = []
		if result_bottom[0][4] != '':

			bottom_list, bottom_info = extract_bottom_lst(result_bottom,df_man, gender)
		else:
			bottom_list= []
			bottom_info = []
	else:
		if result_top[0][4] != '':
			top_list, top_info = extract_top_lst(result_top, df_woman, gender)
		else:
			top_list= []
			top_info = []
		if result_bottom[0][4] != '':
			bottom_list, bottom_info = extract_bottom_lst(result_bottom, df_woman, gender) 
		else:
			bottom_list= []
			bottom_info = []

	startTime = time.time()
	candidate = []
	candidate.extend(top_list)
	candidate.extend(bottom_list)
	# top list = 상의 , bottom_list =하의 리스트 입니다.
	# candidate는 추천 세트 후보들로 위에 2개 리스트인 top_list와 bottom_list를 모두 넣어줍니다.

	top_path ='static/top'
	bottom_path = 'static/bottom'
	result_path = 'static/recommend'
	for path_name in [top_path, bottom_path, result_path]: # 이전 사용자의 시연에 사용된 이미지들은 모두 삭제 top, bottom, recommend 폴더를 비워준다. 
		[os.remove(f) for f in glob.glob(os.path.join(path_name,'*'))]
	logger.info("===") # 아래 3개의 로그는 정말 폴더들이 비워졌는지 체크해주기 위해 사용했습니다.
	logger.info(os.listdir(top_path))
	logger.info(os.listdir(bottom_path))
	logger.info(os.listdir(result_path))
	logger.info("===")
	# 이미지 경로를 생각해야한다. # candidate list는 파일 이름을 갖고 있기에 이를 활용하여 각 이미지 파일의 json 파일에서 mask할 이미지의 폴리곤 좌표와 렉트 좌표를 구해야합니다.
	for i in range(len(candidate)):
		file_name = candidate[i][0] 
		style = candidate[i][1]
		with open('../deepfashion/dataset/'+gender+'/label/'+str(file_name)+'.json',encoding='utf-8-sig')  as json_file:
			#logger.info(file_name, style)
			data =json.load(json_file)
			box_data = {}
			territory = {}
			key_data = list(data['데이터셋 정보']['데이터셋 상세설명']['라벨링'].keys()) # 해당 with open부터 mask_image 함수의 코드를 수정하시고 싶으시다면 먼저 각 파일의 json 안에 중첩 json key들을 보시고 수정하시면 됩니다.

			key = []
			for i in range(1, len(key_data)):
				if data['데이터셋 정보']['데이터셋 상세설명']['라벨링'][key_data[i]] != [{}]:
					if (data['데이터셋 정보']['데이터셋 상세설명']['폴리곤좌표'][key_data[i]] != [{}]):
						key.append(key_data[i])
						box_data[key_data[i]] = []
						box_data[key_data[i]].extend(data['데이터셋 정보']['데이터셋 상세설명']['폴리곤좌표'][key_data[i]][0]['좌표'])
						territory[key_data[i]] = []
						territory[key_data[i]].extend(data['데이터셋 정보']['데이터셋 상세설명']['렉트좌표'][key_data[i]])
			if (box_data != {}):
				mask_images(box_data,key, territory, file_name,style, gender) # 이 코드를 통해 이미지 내에 있는 옷들을 마스킹하고 각각의 옷마다 상의면 static/top, 하의면 static/bottom으로 저장해줍니다.

	img_info = {} # img_base64 dictionary ----------- top
	img_info['top'] = []                  #---------- bottom
	img_info['bottom'] = []               #---------- recommendation # 이 3개는 각 5개의 이미지가 top1 top2 와 같은 형식의 key와 그에 따른 item(base64 이미지) 존재
	img_info['recommend'] = []

	top_lst = os.listdir(top_path) # 마스킹된 이미지들이 저장된 후, 상의 파일명 리스트 추출 
	bottom_lst = os.listdir(bottom_path) # 마스킹된 이미지들이 저장된 후, 하의 파일명 리스트 추출 
 	top_lst, bottom_lst, top_info, bottom_info = checkinfo(top_lst, bottom_lst,top_info, bottom_info, gender) # 상의, 하의마다 누락된 정보가 있는지 확인하고, 있으면 누락된 정보를 채워주는 함수입니다. 

	encodeimages(img_info,'top',top_lst, top_info,bottom_info, gender, result_top) # 이제 상의의 옷들에 대한 url과 style, 그리고 카테고리를 딕셔너리로 저장해주는 코드입니다. img_info['top']에 저장되는 함수입니다. 

	encodeimages(img_info,'bottom',bottom_lst, top_info, bottom_info, gender, result_bottom) # 이제 하의의 옷들에 대한 url과 style, 그리고 카테고리를 딕셔너리로 저장해주는 코드입니다. img_info['bottom']에 저장되는 함수입니다.

	bottom_set = set(bottom_lst) #집합set으로 변환
	bottom_lst = list(bottom_set) #list로 변환 -> duplicate delete 
	top_set = set(top_lst)
	top_lst = list(top_set)

	if len(bottom_lst) < len(bottom_info):
		bottom_flag = [0 for _ in  range(len(bottom_lst))]
	elif len(bottom_lst) >= len(bottom_info):
		bottom_flag = [0 for _ in  range(len(bottom_info))]

	for i in range(len(bottom_flag)):
		if bottom_info[i][1] == '드레스': # 드레스와 같이 상의, 하의의 구분이 없는 옷이면 flag를 1로 저장하고 get_recommendation_list함수에서 flag가 1이면 해당 이미지만 저장해준다. 
			bottom_flag[i] = 1
	result = get_recommendation_set(top_lst, bottom_lst,bottom_flag) # 마지막 결과 출력
	df_woman = df_woman.dropna(how='any') # 데이터 중 현 사용자 값 삭제 (file_path를 NaN으로 줘서 이 방식을 구현함)
	df_man = df_man.dropna(how='any') # 데이터 중 현 사용자 값 삭제 
	logger.info(result)
	logger.info("data for sending")
	
	logger.info("[DONE] print model2_time spent :{:.4f}".format(time.time() - startTime))
	if result != []:
		encodeimages(img_info, 'recommend',result, top_info,bottom_info, gender, result_top)
		logger.info("connect ok")
		logger.info(img_info)
		return make_response(json.dumps(img_info, ensure_ascii=False), 200) # json.dumps
		#return make_response(json.dumps(img_base64, ensure_ascii=False).decode('utf8'))
	else:
		logger.info("connect error")
		return make_response(json.dumps(img_info, ensure_ascii=False), 200)

@app.route('/result', methods=['POST'])
def result():
	startTime = time.time()
	totalTime = time.time()
	path = "./result"
	file_list = os.listdir(path)
	if os.path.exists("result") : 
		for file in os.scandir("result") : 
			os.remove(file.path)
	data = request.files['poto'].read()
	jpg_arr = np.frombuffer(data, dtype=np.uint8)
	img = cv2.imdecode(jpg_arr, cv2.IMREAD_COLOR)
	logger.info("[DONE] print image_load_time spent :{:.4f}".format(time.time() - startTime))
	
	model1Time = time.time()
	predictor = DefaultPredictor(cfg)
	outputs = predictor(img)
	mask_array = outputs['instances'].pred_masks.detach().cpu().numpy()
	num_instances = mask_array.shape[0]
	result = {}
	top ={}
	bottom = {}
	
	lst_name = ['short_sleeved_shirt', 'long_sleeved_shirt', 'short_sleeved_outwear', 'long_sleeved_outwear',
				'vest', 'sling', 'shorts', 'trousers', 'skirt', 'short_sleeved_dress',
				'long_sleeved_dress', 'vest_dress', 'sling_dress'] 

	t = ['short_sleeved_shirt', 'long_sleeved_shirt','short_sleeved_outwear','long_sleeved_outwear','vest',
		'sling', 'short_sleeved_dress', 'long_sleeved_dress', 'vest_dress', 'sling_dress']

	b = ['shorts', 'trousers', 'skirt']
	avg = (960, 540)
	if num_instances > 0:
		logger.info(f'===============total mask : {num_instances}')
		for i, data in enumerate(outputs["instances"].pred_classes.detach().cpu().numpy().tolist()) :
			if lst_name[data] in t : 
				logger.info(f'===============index : {i} category : 상의')
			if lst_name[data] in b : 
				logger.info(f'===============index : {i} category : 하의')
			im = Image.fromarray(img)
			x_min , y_min, x_max, y_max = outputs["instances"].pred_boxes.tensor[i].detach().cpu().numpy()
			x_min = int(x_min)
			x_max = int(x_max)
			y_max = int(y_max)
			y_min = int(y_min)
			coordinate = (int((x_min+x_max)/2), int((y_min+y_max)/2))
			def go_save() : 
				if lst_name[data] in t : 
					top['category'] = '상의'
					top['coordinate'] = str(coordinate)
					category = 'top'
				else : 
					bottom['category'] = '하의'
					bottom['coordinate'] = str(coordinate)
					category = 'bottom'
				mask_array_temp = np.array(0)
				crop_img = im.crop((x_min, y_min, x_max, y_max))
				mask_array_temp = mask_array[i][y_min:y_max, x_min:x_max]
				mask_array_temp = mask_array_temp.reshape((1,) + mask_array_temp.shape)
				mask_array_temp = np.moveaxis(mask_array_temp, 0, -1)
				mask_array_temp = np.repeat(mask_array_temp, 3, axis=2)
				output = np.where(mask_array_temp==False, 255, crop_img)
				img_process = output
				img_process = Image.fromarray(img_process)
				img_process = img_process.convert("RGBA") # 뒷배경 투명을 위해 RGBA로 엽니다. 
				datas = img_process.getdata()
				newData = []
				for item in datas:
					if item[0] > 255 and item[1] > 255 and item[2] > 255:
						newData.append((item[0], item[1], item[2], 0)) # 뒤에꺼는 0으로 넣어줍니다. (결과는 포토샵처럼 나옵니다.)
					else:
						newData.append(item)

				img_process.putdata(newData)
				img_process.save("result/masked_"+category +".png") 
				
			logger.info(f'lst_name={lst_name[data]}')
			if lst_name[data] in t :
				logger.info('top')
				if top.get('category') == None:
					logger.info('top is none, save')
					go_save()
				else:
					old_data_x = int(top.get('coordinate').strip("("")").split(",")[0])
					old_dist = abs(avg[0]-old_data_x)
					new_dist = abs(avg[0]-coordinate[0])
					logger.info(f'old dist={old_dist}, new_dist={new_dist}')
					if old_dist >  new_dist: 
						logger.info('new closer item saved')
						go_save()
					else:
						logger.info('old is more closer')
			else:
				logger.info('bottom')
				if bottom.get('category') == None:
					if coordinate[1] >600 :
						logger.info('bottom is none and coord over 600')
						go_save()
					else : 
						logger.info(f'pass : {coordinate[1]}')
				else:
					if coordinate[1] >600 : 
						old_data_x = int(bottom.get('coordinate').strip("("")").split(",")[0])
						old_dist = abs(avg[0]-old_data_x)
						new_dist = abs(avg[0]-coordinate[0])
						logger.info(f'old dist={old_dist}, new_dist={new_dist}')
						if old_dist >  new_dist: 
							logger.info('new closer item saved')
							go_save()
						else:
							logger.info('old is more closer')
					else : 
						logger.info(f'pass : {coordinate[1]}')
		logger.info("[DONE] print model1_time spent :{:.4f}".format(time.time() - model1Time))
		model2Time = time.time()
		file_list = os.listdir(path)
		logger.info(f'==========file_list :  {file_list}')
		for file_name in file_list : 
			if 'top' in file_name :
				client.publish('top','')
				logger.info(f'category =  top')
			if 'bottom' in file_name :
				client.publish('bottom','')
				logger.info(f'category =  bottom')
			r.set('style', '')
			r.set('detail', '')
			r.set('texture', '')
			s_data = ''
			d_data = ''
			t_data = ''

			temp_dict = {}
			while True:
				s_data = r.get('style')
				if s_data != '': temp_dict['style'] = s_data
				t_data = r.get('texture')
				if t_data != '': temp_dict['texture'] = t_data
				d_data = r.get("detail")			
				if d_data != '': temp_dict['detail'] = d_data
				if len(temp_dict.keys()) == 3 and temp_dict['style'] != '' and temp_dict['detail'] != '' and temp_dict['texture'] != '': break 

			if 'top' in file_name : 
				top['style'] = s_data
				top['detail'] = d_data
				top['texture'] = t_data
			if 'bottom' in file_name : 
				bottom['style'] = s_data
				bottom['detail'] = d_data
				bottom['texture'] = t_data
		logger.info("[DONE] print model2_time spent :{:.4f}".format(time.time() - model2Time))
	
	result['top'] = top
	result['bottom'] = bottom
	if 'category' not in result['top']:
		result['top']['category'] = ''
		result['top']['coordinate'] = '(0,0)'
		result['top']['style'] =''
		result['top']['detail'] = ''
		result['top']['texture'] = ''
	if 'category' not in result['bottom']:
		result['bottom']['category'] = ''
		result['bottom']['style'] =''
		result['bottom']['texture'] =''
		result['bottom']['detail'] =''
		result['bottom']['coordinate'] = '(0,0)'

	logger.info(result)
	logger.info("[DONE] print sendImage_total spent :{:.4f}".format(time.time() - totalTime))
	return make_response(jsonify(result))	

@app.route('/recommendation', methods=['GET'])
def redirect_page():

	return render_template('recommendation.html')

@app.route('/recommendation/image', methods=['GET','POST'])
def recommendation():
	global df_woman
	global df_man
	#params = request.form.to_dict()
	#{'gender': {'gender':'여성'}, 'bottom': {'style':'캐주얼,내츄럴,컨템포러리', 'detail':'팬츠', 'category':'팬츠', 'texture':'팬츠'}, 'top': {'style':'팬츠', 'detail':'팬츠', 'category':'팬츠', 'texture':'팬츠'}}
	param = {'gender': '{gender=여성}', 'bottom': '{style=컨템포러리,캐주얼, detail=스트링, category=팬츠, texture=우븐}', 'top': '{style=매니쉬,캐주얼, detail=드롭숄더, category=셔츠, texture=니트}'}
	#{'gender': '{gender=여성}', 'bottom': '{style=팬츠, detail=팬츠, category=팬츠, texture=팬츠}', 'top': '{style=팬츠, detail=팬츠, category=팬츠, texture=팬츠}'}
	top_category = ['니트웨어','아우터','상의','셔츠','블라우스','베스트','점퍼','티셔츠','드레스','브라탑','후드티','가디건']
	bottom_category = ['팬츠','하의','스커트','청바지','레깅스','조거팬츠',"긴바지"]
	params = param 
	#request.form.to_dict() 
	logger.info(params['gender'].split('=')[1][:-1])
	gender = 'Femaledata' if params['gender'].split('=')[1][:-1] == '여성' else 'Maledata'
	#gender = "Maledata"
	logger.info(gender)
	#print(df_man.collect()[-1])
	result_bottom = []
	result_top = []
	if 'bottom' in params.keys():
		datum = []
		dilim = param['bottom'].replace("style=","/").replace(", detail=","/").replace(", category=","/").replace(", texture=","/").replace("}","/")
		parsed_data = dilim.split("/")
		#print(re.split('{ |style= |, detail= |, category= |, texture= |}',params['bottom']))
		#print(params['bottom'][0:6])
		datum.append(None)
		datum.append('temp')
		datum.append(-1)
		datum.append(parsed_data[2])
		datum.append(parsed_data[4])
		datum.append(convertstyle(parsed_data[1],'eng')) # style 
		datum.append(parsed_data[3])

		#result_bottom = [(None,'트래디셔널',-1, '슬릿',"저지",'트래디셔널,캐주얼,컨템포러리', "긴바지")]
		datum = tuple(datum)
		result_bottom.append(datum)
	
	if 'top' in params.keys():
		datum = []
		dilim = param['top'].replace("style=","/").replace(", detail=","/").replace(", category=","/").replace(", texture=","/").replace("}","/")
		datum.append(None)
		parsed_data = dilim.split("/")
		datum.append('temp')
		datum.append(-1)
		datum.append(parsed_data[2])
		datum.append(parsed_data[4])
		datum.append(parsed_data[1])
		datum.append(parsed_data[3])
		datum = tuple(datum)
		result_top.append(datum)
		#result_top = [(None,'트래디셔널',-1, '슬릿',"저지",'트래디셔널,서브컬쳐,캐주얼', "셔츠")]

	if gender == "Maledata":
		top_list, top_info =extract_top_lst(result_top,df_man, gender)
		bottom_list, bottom_info = extract_bottom_lst(result_bottom,df_man, gender)
	else:
		top_list, top_info = extract_top_lst(result_top, df_woman, gender)
		#print(top_list,"\n",top_info)
		bottom_list, bottom_info = extract_bottom_lst(result_bottom, df_woman, gender) 
	startTime = time.time()
	print(top_list)
	print(bottom_list)
	#print(bottom_list) # 확인용
	candidate = []
	candidate.extend(top_list)
	candidate.extend(bottom_list)
	# top list = 상의 , bottom_list =하의 리스트 입니다.
	# candidate는 추천 세트 후보들로 위에 2개 리스트에서 높은거부터 뽑아서 넣어줍니다.
	# !!!!! 이거 생각해보자. -> top list 와 bottom_list를 같이 합쳐서 진행해도 괜찮을까? 괜찮을거 같다. 어차피 골고루 뽑아서 재조합시키는게 나을거 같다.

	top_path ='static/top'
	bottom_path = 'static/bottom'
	result_path = 'static/recommend'
	for path_name in [top_path, bottom_path, result_path]: # 이전에 사용된 이미지들은 모두 삭제
		[os.remove(f) for f in glob.glob(os.path.join(path_name,'*'))]
	print("===")
	print(os.listdir(top_path))
	print(os.listdir(bottom_path))
	print(os.listdir(result_path))
	print("===")
	# 이미지 경로를 생각해야한다.
	for i in range(len(candidate)):
		file_name = candidate[i][0]
		style = candidate[i][1]
		with open('../deepfashion/dataset/'+gender+'/label/'+str(file_name)+'.json',encoding='utf-8-sig')  as json_file:
			print(file_name, style)
			data =json.load(json_file)
			box_data = {}
			territory = {}
			key_data = list(data['데이터셋 정보']['데이터셋 상세설명']['라벨링'].keys())

			key = []
			for i in range(1, len(key_data)):
				if data['데이터셋 정보']['데이터셋 상세설명']['라벨링'][key_data[i]] != [{}]:
					#print(data['데이터셋 정보']['데이터셋 상세설명']['폴리곤좌표'])
					if (data['데이터셋 정보']['데이터셋 상세설명']['폴리곤좌표'][key_data[i]] != [{}]):
						key.append(key_data[i])
						box_data[key_data[i]] = []
						box_data[key_data[i]].extend(data['데이터셋 정보']['데이터셋 상세설명']['폴리곤좌표'][key_data[i]][0]['좌표'])
						territory[key_data[i]] = []
						territory[key_data[i]].extend(data['데이터셋 정보']['데이터셋 상세설명']['렉트좌표'][key_data[i]])
			if (box_data != {}):
				mask_images(box_data,key, territory, file_name,style, gender)
	print("===")
	print(os.listdir(top_path))
	print(os.listdir(bottom_path))
	print(os.listdir(result_path))
	print("===")
	img_base64 = {} # img_base64 dictionary ----------- top
	img_base64['top'] = []                  #---------- bottom
	img_base64['bottom'] = []               #---------- recommendation # 이 3개는 각 5개의 이미지가 top1 top2 와 같은 형식의 key와 그에 따른 item(base64 이미지) 존재
	img_base64['recommend'] = []
	print("top",top_info)
	print("botoom", bottom_info)
	top_lst = os.listdir(top_path)
	bottom_lst = os.listdir(bottom_path)
	print()
	top_lst, bottom_lst, top_info, bottom_info = checkinfo(top_lst, bottom_lst,top_info, bottom_info, gender)
	print("top",top_info)
	print("botoom", bottom_info)
	encodeimages(img_base64,'top',top_lst, top_info,bottom_info, gender)
	#top_lst = os.listdir(top_path)
	encodeimages(img_base64,'bottom',bottom_lst, top_info, bottom_info, gender)
	#bottom_lst = os.listdir(bottom_path)
	bottom_set = set(bottom_lst) #집합set으로 변환
	bottom_lst = list(bottom_set) #list로 변환
	top_set = set(top_lst)
	top_lst = list(top_set)
	print("Duplicate done bottom", bottom_lst)
	print("Duplicate done top", top_lst)
	#if len(bottom_lst) < len(bottom_info):
	#	bottom_flag = [0 for _ in  range(len(bottom_lst))]
	#elif len(bottom_lst) >= len(bottom_info):
	bottom_flag = [0 for _ in  range(len(bottom_info))]
	print("flag",bottom_flag)
	print(len(bottom_info), len(bottom_lst))
	for i in range(len(bottom_flag)):
		if bottom_info[i][1] == '드레스':
			bottom_flag[i] = 1
	result = get_recommendation_set(top_lst, bottom_lst,bottom_flag) # 마지막 결과 출력
	df_woman = df_woman.dropna(how='any') # 데이터 중 현 사용자 값 삭제 (file_path를 NaN으로 줘서 이 방식을 구현함)
	df_man = df_man.dropna(how='any') # 데이터 중 현 사용자 값 삭제
	print(result)
	print("[DONE] print model2_time spent :{:.4f}".format(time.time() - startTime))
	if result != []:
		encodeimages(img_base64, 'recommend',result, top_info,bottom_info, gender)
		#print(img_base64)
		return make_response(json.dumps(img_base64, ensure_ascii=False)) # json.dumps
		#return make_response(json.dumps(img_base64, ensure_ascii=False).decode('utf8'))
	else:
		return jsonify({"top":[{"title":"블라우스", "context":"모던", "image":"abc"}, {"title":"블라우스", "context":"모던", "image":"abc"},{"title":"블라우스", "context":"모던", "image":"abc"},{"title":"블라우스", "context":"모던", "image":"abc"}],
	 "bottom":[{"title":"블라우스", "context":"모던", "image":"abc"}, {"title":"블라우스", "context":"모던", "image":"abc"},{"title":"블라우스", "context":"모던", "image":"abc"},{"title":"블라우스", "context":"모던", "image":"abc"}],
	 "recommend":["img1","img2","img3"] })

if __name__ == '__main__':
	mq_start()
	app.run(host='0.0.0.0', port=5006, use_reloader=False, threaded=True)
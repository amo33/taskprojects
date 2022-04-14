from curses.ascii import US
from logging import raiseExceptions
import os
from sqlalchemy.exc import IntegrityError
from models import db, User
from flask import send_from_directory, Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import uuid
from PIL import Image
from pathlib import Path
import json
import re
app = Flask(__name__) # 사용할 flask 앱을 가져온다. 
# thumb = Thumbnail(app)
base_dir = os.path.abspath(os.path.dirname(__file__)) # absolute path 지정

img_format = ['jpg','png','peg','jpe']
db_file = os.path.join(base_dir, 'User.sqlite')
file_path = "Term1/sample.json" 
app.config['SQLALCHEMY_DATABASE_URI'] ='sqlite:///' +db_file # 사용할 DB
app.config['SQLALCHEMY_COMMIT_ON_TEARDOWN'] = True # 비즈니스 로직이 끝날때 commit 실행(DB 반영) - 다 처리후 결과 DB에 저장
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False # track은 안한다.

app.config['THUMBNAIL_MEDIA_ROOT'] = '/home/www/media'
app.config['THUMBNAIL_MEDIA_URL'] = '/media/'

app.config['THUMBNAIL_MEDIA_THUMBNAIL_ROOT'] = '/Term1/Image/cache'
app.config['THUMBNAIL_MEDIA_THUMBNAIL_URL'] = '/Image/cache/'
app.config['THUMBNAIL_STORAGE_BACKEND'] = 'flask_thumbnails.storage_backends.FilesystemStorageBackend'
app.config['THUMBNAIL_DEFAUL_FORMAT'] = 'JPEG'

# sercet_file = os.path.join(base_dir, 'secrets.json') # json 파일 위치 
app.config['SECRET_KEY'] ='1234'
db.init_app(app)
db.app = app
db.create_all()

@app.route('/') # flask의 데코레이터로 특정 url에 접속하면 바로 다음 줄에 있는 함수를 호출한다. -> form() 호출 / endpoint is nothing 
def form():
    
    return render_template('form_submit.html', types = 'default')

@app.route('/hello',methods=['GET','POST']) # Use post & Get 
def action():
    ##### Things to do #### 
    ### DB 예외처리!!! - 만약 하나라도 없다면 새로운 페이지로 가서 어느 데이터를 안 넣었는지 알려주고 다시 등록 페이지로 가게 한다. 
    
    if(request.method == 'GET'):
        val = request.args.get('value')
        if val != None: # if we get some parameter after /hello than use the other function
            argu = val[0:1]
            if(val[1:] == 'db'):
                user_info = User.query.filter(User.idx == argu).all()
                print(user_info[0])
                return render_template('form_submit.html', types = 'detail',data = user_info[0])
                
            elif(val[1:] == 'text'):
                f = open('Term1/data.txt', "r")
                lines = f.read()
                new_list = re.split(', |\n', lines)
                temp_user_info = new_list[(int(argu)-1)*4:int(argu)*4]
                user_info = {}
                user_info['idx'] = temp_user_info[0]
                user_info['username'] = temp_user_info[1]
                user_info['age'] = temp_user_info[2]
                user_info['Image'] = temp_user_info[3]
                return render_template('form_submit.html', types = 'detail',data = user_info)
            
        else:
            
            return render_template('form_submit.html', types = 'default') # if db내에 없는 오류가 발생한다면 다시 연결 
    elif(request.method == "POST"):
        
        k = request.files['file'] 
        
        rand_form = uuid.uuid1() # 이미지마다 unique명 유지 
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        age = request.form['age']
        try:
            if(firstname+' '+lastname != ''.join(filter(str.isalnum, firstname)) +' '+''.join(filter(str.isalnum, lastname))):
                raise Exception
            if(age.isdigit() == False):
                raise Exception
        except Exception:
            return render_template('Error_handling.html')
        
        img_dir = str(rand_form)+secure_filename(k.filename)
        """
        try: #이미지 오류는 성공 나머지 값들 오류도 내일 넣어주기 
            if (secure_filename(k.filename[-3:]) not in img_format):
                raise Exception
            if(firstname+lastname == '' or age == ''):
                raise Exception
        except Exception:
            return render_template('Error_Handling.html')
        """
        k.save('Term1/Image/'+img_dir) 
        path = Path('Term1/Image/'+img_dir) #thumbnail 시키기 위한 과정 
        try:
            image = Image.open(path)
            size = (128,128)
            image.thumbnail(size)
            image.save(Path('Term1/static/ThumbedImg/'+img_dir))
        except Exception:
            img_dir = '' # path delete
            pass 
        img_dir = 'static/ThumbedImg/'+ img_dir
        fullname = firstname + ' ' +lastname
        user = User(username = fullname, age = age,Image = img_dir) 
        db.session.add(user)
        try: # used for unique integrity error of image but not going to happen 
            db.session.commit()
        except IntegrityError:
            db.session.rollback()
            return render_template('Error_Handling.html')
        f = open('Term1/data.txt', "a") # 파일에 내용을 write 한다.
        val = User.query.order_by(-User.idx).first()
        lines = [str(val.idx),', ',firstname + ' ' +lastname,', ' ,age, ', ' ,img_dir,'\n']

        f.writelines(lines) 
        f.close()
        return render_template('form_submit.html', types = 'default')
        #return render_template('form_action.html', firstname=firstname, lastname= lastname, age = age)
                #Renders template from the templte folder with the given context 

@app.route('/responsed', methods = ["GET"]) # for ajax & json response between form_submit.html
def responsed():
    #argu = json.loads(request.data)
    #names = argu.get('id')
    id = request.args.get('value')
    user_info = User.query.filter(User.idx == id).all()
    
    user = {
        "Name":user_info[0].username, 
        "Age":user_info[0].age, 
        "Image":user_info[0].Image
        }
    print(1)
    with open(file_path, 'w') as outfile: # !!!!!!!!!!! ㅈㅓ자ㅇ 위치 경로 다시 고치기
        json.dump(user, outfile)
    return jsonify(user)        
    

#db파일과 text 파일은 서로 같은 filepath에 있으면 내 코드상 충돌이 일어날거 같아서 filepath를 애초에 다른것으로 함
@app.route('/detail')
def show_detail():
    return render_template('detail.html')

@app.route('/list', methods= ['GET','POST']) # endpoint is list 
def list_show():
    argu = request.args.get('tag', 'default')
    argu = argu.lower()
    if argu == 'db' :
        return render_template('list.html',types= argu,values = User.query.all())
    elif argu == 'text':
        f = open('Term1/data.txt', "r")
        lines = f.read()
        new_list = re.split(', |\n', lines)
        ids = new_list[0::4]
        names = new_list[1::4]
        ages = new_list[2::4]
        images = new_list[3::4]
        # lines_list = [line.rstrip('\n') for line in lines] -> 이런 식으로 값을 넘기면 문제 발생한다. 
        #print(lines_list)
        return render_template('list.html', types= argu, id = ids ,Name = names, Age = ages, Image = images) # 데이터들 넘기기
    else:
        f = open('Term1/data.txt', "r")
        lines = f.read()
        print(type(lines))
        print(lines)
        new_list = re.split(', |\n', lines)
        print(new_list)
        names = new_list[0::4]
        return render_template('list.html', types = 'default', values = names)

#@app.route('/Image/<regex("([\w\d_/-]+)?.(?:jpe?g|gif|png)"):filename>')
#def media_file(filename):
#    return send_from_directory(app.config['THUMBNAIL_MEDIA_THUMBNAIL_ROOT'], filename)
if __name__ == '__main__':
    #action.counter = 0
    app.run(host='0.0.0.0', debug = True)
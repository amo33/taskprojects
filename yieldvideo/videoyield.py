from flask import Flask, render_template , Response
import cv2 , os
import argparse

app = Flask(__name__)
app.config['SECRET_KEY'] = '1234'

@app.route('/') #html 보여준다.
def index():
    return render_template('index.html')

@app.route('/video_show')
def video_show():
    return Response(yield_video(), mimetype= "multipart/x-mixed-replace; boundary=frame")

def yield_video():
    global video
    if video == 'default':
        capture = cv2.VideoCapture(0)
    else:
        path = 'videosrc/'+ video # video 파일경로 접근
        if os.path.isfile(path) == False:
            capture = cv2.VideoCapture(0)
        else:
            capture = cv2.VideoCapture(path)

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90] #변환 퀄리티 90프로로 지정

    while True:
        ret, frame = capture.read()
        if ret != True:
            break
        frame = cv2.resize(frame,dsize=(0,0),fx=0.6,fy=0.4,interpolation=cv2.INTER_LANCZOS4 )
        frame = cv2.imencode('.jpg',frame, encode_param)[1].tobytes()
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        if cv2.waitKey(1)>0 : break

    print("Video Done")
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Echo websocketClient --movie MOVIE")
    parser.add_argument('--movie', help = "video-directory", default= 'default')
    args = parser.parse_args()
    video = args.movie
    app.run()
#pexels-ekaterina-bolovtsova-7003250.mp4
#production_ID_5077814.mp4
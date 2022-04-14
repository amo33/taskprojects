import argparse
import cv2, os
import socketio
import time 
import simplejpeg
def show_video(video):
    fps =90#pexel = 10 , puppy2 = 60
    pre_time = 0
    sio = socketio.Client()
    sio.connect('http://localhost:5000', wait_timeout=10)
    # 카메라 또는 동영상
    if video == 'default':
        capture = cv2.VideoCapture(0)
    else:
        path = 'videosrc/'+ video # video 파일경로 접근
        #if os.path.isfile(path) == False:
        #    capture = cv2.VideoCapture(0)
        #else:
        #Sample-MP4-Video-File-for-Testing.mp4
        #path
        #SampleTesting.mp4
        # "rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mp4"
        #pexels-anna-bondarenko-5877808.mp4
        #Puppy.mp4
        #Puppy2.mp4
        #SLEEPINGBEAR1.mp4
        #Sequence_09_2.mov
        #cat.mp4
        #lion.mp4
        capture = cv2.VideoCapture(path)

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80] #변환 퀄리티
    #fps = capture.get(cv2.CAP_PROP_FPS) 
    while True:
        ret, frame = capture.read()
        if ret != True:
            break
        current_time = time.time() - pre_time
        if (ret is True) and (current_time>1/fps):
            pre_time = time.time()
            # todo frame drop and opencv 로 영상 fps 확인 
            frame = cv2.resize(frame, dsize=(700,500),interpolation=None)
            #frame = cv2.resize(frame,dsize=(0,0),fx=0.5,fy=0.5,interpolation=cv2.INTER_LANCZOS4)
            frame = cv2.imencode('.jpg',frame, encode_param)[1].tobytes()
            #frame = simplejpeg.encode_jpeg(frame, 90,colorspace='BGR') # 비효율적 
            sio.emit('streaming', frame)
       
        #if cv2.waitKey(1)>0 : break #checking waitkey makes the frame slow
    print("Video Done")
    sio.disconnect()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Echo websocketClient --movie MOVIE")
    parser.add_argument('--movie', help = "video-directory", default= 'default')
    args = parser.parse_args()
    show_video(video = args.movie)



#------예시 영상 파일 ------
#pexels-ekaterina-bolovtsova-7003250.mp4
#production_ID_5077814.mp4
#SampleTesting.mp4
#sample-5s.mp4

#-------전송 종류 -------
#frame = base64.b64encode(frame).decode('utf-8')
#1 data = 'data:image/jpeg;base64,' +frame #이미지 자체 넘겨주는 거면 data: 라고 라벨해줘야한다.
#2 frame = binascii.hexlify(frame).decode('utf-8') 
#1 data = frame
#data = b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
#base64 이미지 자체 넘겨주는 거면 data: 라고 라벨해줘야한다.
#data = b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + base64.b64decode(frame) + b'\r\n\r\n'
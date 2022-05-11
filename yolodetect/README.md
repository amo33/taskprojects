### 리눅스 현재 폴더에 존재하는 모든 하위 폴더 내의 파일들 이동 
find -type f -execdir mv "{}" ../ \;
### yolov5 & pytorch 
I've done this training in aws ec2 environment. 
All the settings are done by googling and the task was to identify black car, white car, and other cars. 
I used labelImg to do the labeling. (not Roboflow)

### Things I need to memo. 
#### 1. scp 
        scp -r files client@address:/<directory> means send local files to server. 
        scp client@address:/<directory>/<filename> means download file from server. 
#### 2. git & sourcetree setting for contribution map 
        https://wellbell.tistory.com/43
        https://velog.io/@starkdy/GitHub-%EC%9E%94%EB%94%94-%EC%98%A4%EB%A5%98-%ED%95%B4%EA%B2%B0-%EB%B0%A9%EB%B2%95
## 공부용
https://github.com/ultralytics/yolov5/wiki/Tips-for-Best-Training-Results
https://github.com/robmarkcole/yolov5-flask/blob/master/templates/index.html

### docker 접속 

triplet_tank@ip-172-31-20-14 ~]$ docker cp /home/triplet_tank/deep_fashion/. 8b0f58124f48:/deepfashion/
이 명령어는 바로 접속 후 컨테이너 실행 후  치면된다. 8b0~은 컨테이너 번호다.
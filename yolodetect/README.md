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
## 공부용
https://github.com/ultralytics/yolov5/wiki/Tips-for-Best-Training-Results
https://github.com/robmarkcole/yolov5-flask/blob/master/templates/index.html
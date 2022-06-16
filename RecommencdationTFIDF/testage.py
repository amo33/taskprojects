import cv2 
import requests 
import base64 
import json 
print(cv2.__version__)
cap = cv2.VideoCapture(1)
for i in range(10):
    
    if cap.isOpened():
        while True:
            ret, fram = cap.read()
            if ret:
                cv2.imshow("camera",fram) #프레임 이미지 표시
                key = cv2.waitKey(1000)
                #if cv2.waitKey(1) != -1:
                if key == -1:
                    cv2.imwrite("./photo.jpg",fram)
                    break
            else:
                print("no fram")
                break
    else:
        print("can't open camera")
    with open('./photo.jpg', 'rb') as img:
        base64_string = base64.b64encode(img.read())
    datum = {
       "img" : base64_string
    }
    url = 'http://175.209.155.106:5005/get_man_info'
    
    response = requests.post(url, data= datum)
    

    print(response.text)
cap.release()
cv2.destroyAllWindows()
'''
import cv2 
import requests 
import base64 
import json 
print(cv2.__version__)
clicked = False 
def mouse_click(event):
   global clicked 
   if event == cv2.EVENT_LBUTTONDBCLK:
      clicked = True
   return clicked  
cap = cv2.VideoCapture(0)
ret, fram = cap.read()
for i in range(10):
    clicked = False 
    
    if cap.isOpened():
        while True:
            ret, fram = cap.read()
            if ret:
                cv2.imshow("camera",fram) #프레임 이미지 표시
                cv2.namedWindow('frame')
                #if cv2.waitKey(1) != -1:
                if cv2.setMouseCallback('frame', mouse_click) == True:
                    cv2.imwrite("./photo.jpg",fram)
                    break
            else:
                print("no fram")
                break
    else:
        print("can't open camera")
    with open('./photo.jpg', 'rb') as img:
        base64_string = base64.b64encode(img.read())
    datum = {
       "img" : base64_string
    }
    url = 'http://175.209.155.106:5005/get_man_info'
    
    
    response = requests.post(url, data= datum)
    

    print(response.text)
cap.release()
cv2.destroyAllWindows()
'''
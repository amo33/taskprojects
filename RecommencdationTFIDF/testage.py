import cv2 
print(cv2.__version__)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read() # ret is true if reading 
    if not ret:
        print("Can't receive")
        break 
    cv2.imshow("video_check",frame)

    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
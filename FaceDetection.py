import cv2

alg = "haarcascade_frontalface_alt.xml" #access the xml file available in python file location ->Lib->site-packages->cv2->data
haar_cascade = cv2.CascadeClassifier(alg) #loading the model
cam = cv2.VideoCapture(1) #initializing camera. If not initialized, change the number between 0 to 10.

while True:
    _,img = cam.read() #read frame from camera
    grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #converting to grayscale image
    face = haar_cascade.detectMultiScale(grayImg,1.3,4)

    for (x,y,w,h) in face:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow("FaceDetection",img)
    key = cv2.waitKey(10)
    if key == ord("q"):
        break
cam.release()
cv2.destroyAllWindows()

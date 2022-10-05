from tensorflow.keras.models import load_model
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
model = load_model('model gestur.h5')

offset = 20
imgSize = 100

counter = 0
labels = ["BAD","CALL","GOOD","MUTE","PEACE"]

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgCrop = img[y:y+w,x:x+w]


        aspectRatio = h/w
        label = 0
        if aspectRatio > 1:
            imgResize = cv2.resize(imgCrop, (100,100))
            normalized=imgResize/255.0
            reshaped=np.reshape(normalized,(1,100,100,3))
            prediction = model.predict(reshaped)
            label=np.argmax(prediction,axis=1)[0]
        cv2.rectangle(imgOutput,(x,y),(x+w,y+h),(0,0,0),2)
        cv2.rectangle(imgOutput,(x,y-40),(x+w,y),(0,0,0),-1)
        cv2.putText(imgOutput,labels[label],(x,y-20),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
        
    cv2.imshow("Image",imgOutput)
    key=cv2.waitKey(1)
    
    if(key==27):
        break
cv2.destroyAllWindows()
cap.release()
        
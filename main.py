import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import*
import cvzone


model=YOLO('yolov8s.pt')

area1=[[455,194],[545,194],[545,220],[455,220]]
area2=[[455,220],[545,220],[545,246],[455,246]]

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        point = [x, y]
        print(point)
  
        

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)
cap=cv2.VideoCapture("C:/Users/ravje/Downloads/IMG_4361.MOV")



count=0
persondown={}
tracker=Tracker()
counter1=[]

personup={}
counter2=[]
cy1=194
cy2=220
offset=6
while True:    
    ret,frame = cap.read()
    if not ret:
        break
#    frame = stream.read()

    count += 1
    if count % 2 != 0:
        continue
    frame=cv2.resize(frame,(1020,500))
   

    results=model.predict(frame, verbose=False)
 #   print(results)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
#    print(px)
    list=[]
   
    for index,row in px.iterrows():
#        print(row)
 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        cls_id=int(row[5])
        
        if cls_id==0:
            cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,255),2)
            cv2.putText(frame,f'person',(x1,y1),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,255),1)
       
        
        
    cv2.polylines(frame,[np.array(area1, np.int32)], True, (0,255,0),2)
    cv2.putText(frame, str("1"), (466, 485), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)

    cv2.polylines(frame,[np.array(area2, np.int32)], True, (0,255,0),2)
    cv2.putText(frame, str("2"), (466, 485), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)
   

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()
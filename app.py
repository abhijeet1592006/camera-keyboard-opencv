import pyautogui as p
import cv2
import math as m

p.FAILSAFE=False
import mediapipe as mp
import numpy as np
repeat=300
mphands=mp.solutions.hands

hands=mphands.Hands()

draw=mp.solutions.drawing_utils


video=cv2.VideoCapture(0)
low=np.array([40,40,40])
upper=np.array([90,255,255])
lowery=np.array([20,100,100])
uppery=np.array([30,255,255])

while True:

    ok,frame=video.read()
    


    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(hsv,low,upper)
    mask2=cv2.inRange(hsv,lowery,uppery)

    fr=cv2.bitwise_and(frame,frame,mask=mask)

    result=hands.process(rgb)

    contor,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contor2,_2=cv2.findContours(mask2,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    ox=0
    oy=0
    ow=0
    oh=0
    ocy=0
    ocx=0
    yellow_cy=0
    yellow_cx=0
    for i in contor:
        if(cv2.contourArea(i))>600:
            ox,oy,ow,oh=(cv2.boundingRect(i))
            ocy=(oy+oh//2)
            ocx=ox+ow//2
            cv2.putText(frame,("A"),(ox-10,oy-10),cv2.FONT_HERSHEY_DUPLEX,2,(0,0,255),2)
            cv2.rectangle(frame,(ox,oy),((ow+ox),(oh+oy)),(255,0,0),(3))
    for i in contor2:
        if(cv2.contourArea(i))>600:
            yellow_x,yellow_y,yellow_w,yellow_h=(cv2.boundingRect(i))
            yellow_cy=(yellow_y+yellow_h//2)
            yellow_cx=(yellow_x+yellow_w//2)
            cv2.putText(frame,("DEL"),(yellow_x-10,yellow_y-10),cv2.FONT_HERSHEY_DUPLEX,1,(0,0,255),2)
            cv2.rectangle(frame,(yellow_x,yellow_y),((yellow_w+yellow_x),(yellow_h+yellow_y)),(255,0,0),(3))


    if result.multi_hand_landmarks:
        for i in result.multi_hand_landmarks:
            for id,pos in enumerate(i.landmark):
                y,x,c=frame.shape

                cx=int(x*(pos.x))
                cy=int(y*(pos.y))
                # if id==8:  ##it is creating lag on my lowend system
                #     p.moveTo(max(20,cx*5),max(20,cy*5))
                
                distance1=m.sqrt((ocx-cx)**2+(ocy-cy)**2)
                distance2=m.sqrt((yellow_cx-cx)**2+(yellow_cy-cy)**2)
                draw.draw_landmarks(frame,i,mphands.HAND_CONNECTIONS)
                if id==8:
                    if (distance1<50):
                        cv2.circle(frame,(cx,cy),(10),(0,255,0),(4))
                        if(repeat>80):
                            p.typewrite("A")
                            repeat=0
                            
                    if(distance2<50):
                        cv2.circle(frame,(cx,cy),(10),(0,255,0),(4))
                        if(repeat>80):
                            p.typewrite(["backspace"])
                            repeat=0
                            

                    repeat=repeat+20
                    
                
                    
                
    cv2.imshow("frame",frame)

    if(cv2.waitKey(1))==ord("q"):
        break


cv2.destroyAllWindows()

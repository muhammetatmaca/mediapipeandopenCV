import cv2
import time
import mediapipe as mp

cap =cv2.VideoCapture(0)

mpHand=mp.solutions.hands

hands=mpHand.Hands(max_num_hands=3)

mpDraw=mp.solutions.drawing_utils

pTime = 0
cTime = 0



while True:
    success,frame=cap.read()
    frame_rgb=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    result=hands.process(frame_rgb)
    
    print(result.multi_hand_landmarks)
    
    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame,handLms,mpHand.HAND_CONNECTIONS)
            
            
            for id, lm in enumerate(handLms.landmark):
              # print(id, lm)
              h, w, c = frame.shape
              
              cx, cy = int(lm.x*w), int(lm.y*h) 
              
              # bilek
              if id == 4:
                  cv2.circle(frame, (cx,cy), 9, (255,0,0), cv2.FILLED)
            
            
            
            # for id,lm in enumerate(handLms.landmark):
            #      print(id,lm)
                 
            
            # h,w,c=frame.shape
            
            # cx,cy=int(lm.x*w),int(lm.y*h)
            
            # if id==0:
            #     cv2.circle(frame,(cx,cy),9,(255,0,0),cv2.FILLED)
            
    
    cTime = time.time()
    fps = 1 / (cTime- pTime)
    pTime = cTime
    
    cv2.putText(frame, "FPS: "+str(int(fps)), (10,75), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 5)
    
    cv2.imshow("img",frame)
    
    cv2.waitKey(1)
    
    
    
    
cap.release()
cv2.destroyAllWindows()
        

import cv2
import mediapipe as mp
import time 
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils
psList=[]

pTime=0


cap = cv2.VideoCapture("livakovic.mp4") 
cap.set(3, 640)
cap.set(4, 480)

while True:
    success, frame = cap.read()

    if not success:
        print("Video okunamadı veya bitti")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(frame_rgb)
    print(result.pose_landmarks)

    if result.pose_landmarks:
        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        for id ,lm in  enumerate(result.pose_landmarks.landmark):
            h, w, _ = frame.shape
            cx, cy = int(lm.x*w), int(lm.y*h)
            psList.append([id, cx, cy])
            if id ==17:
                cv2.circle(frame, (cx,cy), 15,(0,255,0),cv2.FILLED)
            if id ==18:
                    cv2.circle(frame, (cx,cy), 15,(0,255,0),cv2.FILLED)
         
    
        
    cTime=time.time()
    FPS=1/(cTime-pTime)
    pTime=cTime
    
    cv2.putText(frame, str(int(FPS)) ,(10,65),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),2)
    
    
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q")&32: 
        cv2.destroyAllWindows()
        break
   



    
    



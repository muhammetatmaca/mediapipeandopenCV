import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

mpHand = mp.solutions.hands
hands = mpHand.Hands()
mpDraw = mp.solutions.drawing_utils

tipIds = [4, 8, 12, 16, 20]
cTime = 0
pTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        finger_counts = []
        for hand_num, handLms in enumerate(results.multi_hand_landmarks):
            mpDraw.draw_landmarks(img, handLms, mpHand.HAND_CONNECTIONS)

            lmList = []
            for id, lm in enumerate(handLms.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

            fingers = []

            # Baş parmak
            if lmList[tipIds[0]][2] < lmList[tipIds[0] - 1][2]:
                fingers.append(1)
            else:
                fingers.append(0)

            # Diğer 4 parmak
            for id in range(1, 5):
                if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            totalF = fingers.count(1)
            finger_counts.append(totalF)

        for hand_num, fingers in enumerate(finger_counts):
            cv2.putText(img, f"Hand {hand_num + 1}: {fingers}", (30, 125 + 50 * hand_num),
                        cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 8)
        cv2.putText(img, f"Total Hands: {len(finger_counts)}", (200, 400),
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 5)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, "FPS: " + str(int(fps)), (100, 450),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 5)

    cv2.imshow("img", img)
    cv2.waitKey(1)


cap.release()
cv2.destroyAllWindows()

    

    
    

    
    
    
    
    

    
    
    
    
    

    
    
    
    
    

    
    
    

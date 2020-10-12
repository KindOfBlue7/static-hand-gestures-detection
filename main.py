import numpy as np
import cv2

# Control variables

ip = "192.168.1.4:4747"
link = "http://" + ip + "/video"
font = cv2.FONT_HERSHEY_SIMPLEX

cap = cv2.VideoCapture(link)
img_w = cap.get(3)
img_h = cap.get(4)

# Checking whether camera is intialized

if cap.isOpened() == False:
    cap.open(0)

print("Width: %d\nHeight: %d" % (img_w, img_h))

while(cap.isOpened()):
    # Capturing frame by frame
    ret, frame = cap.read()
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    frame = cv2.putText(hsv,
                'Press Q to quit',
                (50,50),
                font,
                1,
                (255,255,255),
                 2,
                 cv2.LINE_AA)
    
    cv2.imshow('frame', frame)
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
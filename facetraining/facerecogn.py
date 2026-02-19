from ultralytics import YOLO
import cvzone
import cv2
import math




# Running real time from webcam
cap = cv2.VideoCapture(0)
model = YOLO('Kface.pt')


# Reading the classes
classnames = ['Khaarl ']

while True:
    ret,frame = cap.read()
    frame = cv2.resize(frame,(640,480))
    result = model(frame,stream=True)

    # Getting bbox,confidence and class names informations to work with
    for info in result:
        boxes = info.boxes
        for box in boxes:
            confidence = box.conf[0]
            confidence = math.ceil(confidence * 100)
            Class = int(box.cls[0])
            
            # Print the confidence to the terminal so you can see what the AI is thinking!
            print(f"AI saw something! Confidence: {confidence}%")

            # Lowered the threshold to 20 to test if the model is just unsure
            if confidence > 20:
                x1,y1,x2,y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),5)
                
                # Removed the extra space in 'Khaarl ' from your classnames list above if you update it
                cvzone.putTextRect(frame, f'{classnames[Class]} {confidence}%', [x1 + 8, y1 + 100],
                                   scale=1.5,thickness=2)

    cv2.imshow('frame',frame)
    cv2.waitKey(1)
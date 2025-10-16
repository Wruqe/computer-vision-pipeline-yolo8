from ultralytics import YOLO
import cv2




model = YOLO("yolov8n-pose.pt")
cap = cv2.VideoCapture(0)





while True:
    ret, frame = cap.read()
    if not ret:
        break
    

    results = model(frame)
    annotated = results[0].plot()
    cv2.imshow("Pose Test", annotated)
    
    
    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()
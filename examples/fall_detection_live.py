import cv2 
from ultralytics import YOLO
from fall_heuristics import fall_score



model = YOLO("yolov8n-pose.pt")


cap = cv2.VideoCapture(0)



while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model(frame)
    result = results[0]
    boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else []
    keypoints = result.keypoints.data.cpu().numpy() if result.keypoints is not None else []
    
    for i in range(len(boxes)):
        box = boxes[i]
        kp = keypoints[i]
        
        score = fall_score(kp, box)
        
        color = (0, 0, 255) if score > 0.6 else (0, 255, 0)  # red = danger, green = safe
        
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1,y1),(x2, y2), color, 2)
        
        cv2.putText(frame, f"Fall risk:{score:.2f}", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        
        cv2.imshow("Fall Detection", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
cap.release()
cv2.destroyAllWindows()
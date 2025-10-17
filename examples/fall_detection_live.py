import cv2 
from ultralytics import YOLO
import argparse
import time
import os
from datetime import datetime
from fall_heuristics import fall_score





model = YOLO("yolov8n-pose.pt")
EVIDENCE_DIR = "examples/fall_evidence"
os.makedirs(EVIDENCE_DIR, exist_ok=True)



fall_event_active = False
fall_event_time = 0.0
last_score = 0

## cli commands
def parse_args():
    parser = argparse.ArgumentParser(description="AI Fall Detection - YOLOv8 + Pose")
    parser.add_argument("--video", type=str, default=None,
                        help="Path to video file. If not set, webcam is used.")
    parser.add_argument("--save", action="store_true",
                        help="Save output video with annotations")
    parser.add_argument("--output", type=str, default="output_fall_detection.mp4",
                        help="Output video file name when using --save")
    return parser.parse_args()

args = parse_args()



def export_screenshot(fall_frame):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = os.path.join(EVIDENCE_DIR, f"fall_{timestamp}.png") 
    cv2.imwrite(filename, fall_frame)
    
    
    
    
if args.video:
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise ValueError(f"cannot open {args.video}")
    print(f"Running capture on {args.video}")
else:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise ValueError(f"cannot open webcam")
    print("running on webcam")
    
    
writer = None

if args.save:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    print(f"💾 Saving output video to {args.output}")


while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model.predict(frame, verbose=False)
    result = results[0]
    boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else []
    keypoints = result.keypoints.data.cpu().numpy() if result.keypoints is not None else []
    no_person_detected = len(boxes) == 0
    
    if no_person_detected and last_score > 0.5:
        cv2.putText(frame, "🚨 FALL DETECTED (Lost Person)", (40, 40),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2)
        print( "🚨 FALL DETECTED (Lost Person)")
        export_screenshot(frame)



    for i in range(len(boxes)):
        box = boxes[i]
        kp = keypoints[i]
        x1, y1, x2, y2 = map(int, box)

        score = fall_score(kp, box)
        last_score = score
        
        if score >= .5:
            if not fall_event_active:
                fall_event_active = True
                fall_event_time = time.time()
            else:
                
                if time.time() - fall_event_time > 0.5:
                    cv2.putText(frame, "Fall Detected!", (x1, y1 - 40))
                    print( "🚨 FALL DETECTED")
                    export_screenshot(frame)

        else:
            # If we briefly lose the person after a high score, keep event active a little longer
             if fall_event_active and (time.time() - fall_event_time < 1.0):
                pass  # keep event alive temporarily
             else:
                fall_event_active = False
        
        color = (0, 0, 255) if score > 0.6 else (0, 255, 0)  # red = danger, green = safe
        
        cv2.rectangle(frame, (x1,y1),(x2, y2), color, 2)
        
        cv2.putText(frame, f"Fall risk:{score:.2f}", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        
    cv2.imshow("Fall Detection", frame)
        
    if writer:
         writer.write(frame)

        # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
    

if writer:
    writer.release()
cap.release()
cv2.destroyAllWindows()



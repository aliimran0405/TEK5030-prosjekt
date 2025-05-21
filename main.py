import cv2
import os

from ultralytics import YOLO
from ball_tracker import *

# Path for where you store your input vids
INPUT_VID_DIR_PATH = "./input_vid/" 




# Read input directory to get list of all input vids and ask user for choice of vid (can also set path if you only have one vid)
inp_vids = os.listdir(INPUT_VID_DIR_PATH)
print('\n', inp_vids, '\n')
vid_idx = int(input("Which video would you like to process? (Enter idx) \n").strip())

#model = YOLO('yolov8m.pt') #yolov8m.pt / yolo11n.pt

ball_tracker = BallTracker()

# Init capture object for video file
cap = cv2.VideoCapture(INPUT_VID_DIR_PATH + inp_vids[vid_idx])


if (not cap.isOpened()):
    raise Exception("Could not find video stream")

FRAME_SKIP = 5
THRESHOLD = 0.35
frame_count = 0
TRACKING = True

# Read video file frame for frame until finished
while (cap.isOpened()):
    ret, frame = cap.read()

    if ret == True:
        frame = cv2.resize(frame, (640, 640))

        #annotated_frame = ball_tracker.process_frame(frame)

        
        if frame_count % FRAME_SKIP == 0:
            annotated_frame = ball_tracker.process_frame(frame)
        else:
            annotated_frame = frame
        

        # for r in results:
        #     for box in r.boxes:
        #         cls = int(box.cls[0])
        #         if cls == 32:  # Class 32 is reserved for a football
        #             xyxy = box.xyxy[0].cpu().numpy().astype(int)
        #             conf = float(box.conf[0])
        #             label = f"{model.names[cls]} {conf:.2f}"
        #             cv2.rectangle(annotated_frame, tuple(xyxy[:2]), tuple(xyxy[2:]), (0, 255, 0), 2)
        #             cv2.putText(annotated_frame, label, tuple(xyxy[:2]), cv2.FONT_HERSHEY_SIMPLEX,
        #                         0.5, (0, 255, 0), 2)

        # for r in results:
        #     for box in r.boxes:
        #         cls = int(box.cls[0])
        #         conf = float(box.conf[0])
        #         print(f"Detected class {cls} with confidence {conf:.2f}")

        
        cv2.imshow('Ball Tracking', annotated_frame)
        # Press q to quit
        if cv2.waitKey(int(25 / FRAME_SKIP)) & 0xFF == ord('q'):
            break


    else:
        break



# When done reading video file: release capture object and close all cv2 windows.
cap.release()
cv2.destroyAllWindows()
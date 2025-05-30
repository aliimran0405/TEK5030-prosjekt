import cv2
import os
from ultralytics import solutions, YOLO
import torch
import time
import numpy as np
from ball_tracker import BallTracker
from synne import PlayerTracker


INPUT_VID_PATH = "./input_vid/fotball2.mov" # f.ex: "./some_dir/some_vid.mov"

cap = cv2.VideoCapture(INPUT_VID_PATH)
if (not cap.isOpened()):
    raise Exception("Could not find video stream")


# Initialize the videowriter to match original video stream
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("instance_segmentation_result_.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps=fps, frameSize=(640,640))

model = YOLO(model="./models/yolo11m-seg.pt")

# Create instances of player and ball tracker classes
ball_tracker = BallTracker()
player_tracker = PlayerTracker(model)

# Read video file frame for frame until finished
n_frames = 0
while (cap.isOpened()):
    ret, frame = cap.read()

    if ret == True:
        frame = cv2.resize(frame, (640, 640)) # Resize frame

        # Call 'process_frame' methods of each tracker class
        results_ball = ball_tracker.process_frame(frame)
        results_player = player_tracker.process_frame(frame)
        result = cv2.addWeighted(results_ball, 0.5, results_player, 0.5, 0) # Combine the two processed frames and show/write to file
        video_writer.write(result)
        cv2.imshow("TRACKER", result)

        # Press q to quit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        
        n_frames += 1
        if n_frames > 7*fps: #Only run for a certain amount of frames (set to seconds_wanted*fps of input stream)
            break

    else:
        break



# When done reading video file: release capture object and close all cv2 windows.
cap.release()
video_writer.release()
cv2.destroyAllWindows()
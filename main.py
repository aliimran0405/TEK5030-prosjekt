import cv2
import os
from ultralytics import solutions, YOLO
import torch
import time


# Path for where you store your input vids
INPUT_VID_DIR_PATH = "./input_vid/" 




# Read input directory to get list of all input vids and ask user for choice of vid (can also set path if you only have one vid)
inp_vids = os.listdir(INPUT_VID_DIR_PATH)
print('\n', inp_vids, '\n')
vid_idx = int(input("Which video would you like to process? (Enter idx) \n").strip())


# Init capture object for video file
cap = cv2.VideoCapture(INPUT_VID_DIR_PATH + inp_vids[0])
if (not cap.isOpened()):
    raise Exception("Could not find video stream")


# Video writer
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("instance_segmentation_result.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps=fps, frameSize=(w,h))

isegment = solutions.InstanceSegmentation(show=False,
                                          model="yolo11n-seg.pt",
                                          tracker="botsort.yaml",
                                          classes=[0],
                                          device='cuda' if torch.cuda.is_available() else 'cpu'
                                          )


# Read video file frame for frame until finished
while (cap.isOpened()):
    ret, frame = cap.read()

    if ret == True:
        #cv2.imshow('Frame', frame)
        results = isegment(frame)
        video_writer.write(results.plot_im)

        # Press q to quit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break


    else:
        break



# When done reading video file: release capture object and close all cv2 windows.
cap.release()
video_writer.release()
cv2.destroyAllWindows()
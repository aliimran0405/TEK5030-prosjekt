import cv2
import os

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



# Read video file frame for frame until finished
while (cap.isOpened()):
    ret, frame = cap.read()

    if ret == True:
        cv2.imshow('Frame', frame)

        # Press q to quit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break


    else:
        break



# When done reading video file: release capture object and close all cv2 windows.
cap.release()
cv2.destroyAllWindows()
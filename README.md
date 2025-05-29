# TEK5030 - Project - Football Tracker

## Features

- A BallTracker class with a method to process individual frames and track a football from video footage using object detection.
- A PlayerTracker class that processes individual frames and uses instance segmentation to detect players on the field and split them into teams.
- A main script that extracts frames from a video source using OpenCV and merges the results from Player- and Ball-Tracker then writes to a videofile.

### Install dependencies:

#### Windows
```bash
py -m pip install -r requirements.txt
```

#### Unix/macOS
```bash
python3 -m pip install -r requirements.txt
```

### Usage

Set INPUT_VID_DIR_PATH variable in 'main' to path of your input video dir, then run 'main'.

```bash
python main.py
```

import cv2
import json
from ultralytics import YOLO

'''Simple ball tracker class that handles the behavior of the ball tracker'''
class BallTracker():
    def __init__(self, model_path='./models/best.pt', confidence_threshold=0.10):
        self.tracker_model = YOLO(model_path)
        self.threshold = confidence_threshold
        self.ball_class_id = 0  # Class ID for football in YOLO at default 32 / 0 for self-trained model
        self.ball_position = [] # Stores the ball's position (frame_num, pos)
        self.current_frame_num = 0

    def process_frame(self, frame):

        ball_detected = False # Flag for checking ball detection

        results = self.tracker_model(frame, classes=[self.ball_class_id], verbose=False) # retrieve results from the model (only class 0 results)
        annotated_frame = frame.copy()
        
        # loop through all results (all frames)
        for r in results:
            # find all bounding boxes in current frame
            for box in r.boxes:
                conf = float(box.conf[0])
                # make sure confidence is high enough to show result
                if conf >= self.threshold:
                    xyxy = box.xyxy[0].cpu().numpy().astype(int) # retrieve (x1, y1, x2, y2) positions of the ball/bounding box
                    label = f"{self.tracker_model.names[self.ball_class_id]} {conf:.2f}" # get name of class and confidence
                    
                    # draw bounding box and add text in frame
                    cv2.rectangle(annotated_frame, tuple(xyxy[:2]), tuple(xyxy[2:]), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, label, tuple(xyxy[:2]), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 2)
                    
                    
                    center = self._get_center(xyxy)

                    # save ball positions for testing
                    self.ball_position.append({
                        'frame_num': self.current_frame_num,
                        'x': center[0],
                        'y': center[1],
                        'confidence': conf
                    })
                    
                    ball_detected = True
        
        # also save 'non-detected' positions
        if ball_detected == False:
            self.ball_position.append({
                'frame_num': self.current_frame_num,
                'x': None,
                'y': None,
                'confidence': None
            })
        self.current_frame_num += 1
        
        # Only temporary/for testing (reviewing the positions of the ball each frame)
        with open('ball_positions.json', 'w') as f:
            json.dump(self.ball_position, f, indent=2, default=str)
        
        return annotated_frame

    # Gets the center of the bounding box (uses xyxy method)
    # Not really used in the end...
    def _get_center(self, bbox):
        x1, x2, y1, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    
   

    
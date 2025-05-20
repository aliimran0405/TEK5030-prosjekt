# ball_tracker.py
import cv2
import json
from ultralytics import YOLO

class BallTracker():
    def __init__(self, model_path='yolo11n.pt', confidence_threshold=0.35):
        self.tracker_model = YOLO(model_path)
        self.threshold = confidence_threshold
        self.ball_class_id = 32  # Class ID for football in YOLO at default / 0 for self-trained model
        self.trajectory_points = []
        self.ball_position = [] # Stores the ball's position (frame_num, pos)
        self.current_frame_num = 0

    def process_frame(self, frame):

        ball_detected = False

        results = self.tracker_model(frame, classes=[self.ball_class_id], verbose=False)
        annotated_frame = frame.copy()
        
        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf >= self.threshold:
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    label = f"{self.tracker_model.names[self.ball_class_id]} {conf:.2f}"
                    
                    # Draw bounding box
                    cv2.rectangle(annotated_frame, tuple(xyxy[:2]), tuple(xyxy[2:]), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, label, tuple(xyxy[:2]), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 2)
                    
                    
                    center = self._get_center(xyxy)

                    self.ball_position.append({
                        'frame_num': self.current_frame_num,
                        'x': center[0],
                        'y': center[1],
                        'confidence': conf
                    })
                    #self.trajectory_points.append(center)
                    #self._draw_trajectory(annotated_frame)
                    ball_detected = True
        
        if ball_detected == False:
            self.ball_position.append({
                'frame_num': self.current_frame_num,
                'x': None,
                'y': None,
                'confidence': None
            })
        self.current_frame_num += 1
        
        # Only temporary/for testing 
        with open('ball_positions.json', 'w') as f:
            json.dump(self.ball_position, f, indent=2, default=str)
        
        return annotated_frame

    # Gets the center of the bounding box (uses xyxy method)
    def _get_center(self, bbox):
        x1, x2, y1, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    
   

    
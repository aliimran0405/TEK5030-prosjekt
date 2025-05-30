from ultralytics import YOLO

model = YOLO('yolov8m.pt')

# Dataset downloaded from: https://universe.roboflow.com/indian-institute-of-science/football-tracker/dataset/1

model.train(
    data='football_dataset/data.yaml', 
    epochs=20,                         
    imgsz=640,                         
    batch=16,                          
    name='football_yolo',             
    patience=10,                       
    device='cpu'                         
)
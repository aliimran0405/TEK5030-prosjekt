from ultralytics import YOLO

model = YOLO('yolov8x')

results = model.predict('input_vid/frankrike_clip.mov', save=True)

print("RESULTS: ", results)
print("=====================================")
for box in results[0].boxes:
    print(box)


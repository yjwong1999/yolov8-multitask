import sys
sys.path.insert(0, "/home/tham/Desktop/YOLOv8-multi-task-modified/ultralytics")

from ultralytics import YOLO


number = 2 #input how many tasks in your work
model = YOLO('/home/tham/Desktop/YOLOv8-multi-task-modified/ultralytics/runs/multi/yolopm/weights/best.pt')  # Validate the model
model.predict(source='/home/tham/Desktop/red.jpg', name='try_predict', save=True, conf=0.25, iou=0.45, show_labels=True)

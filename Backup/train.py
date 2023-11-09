import sys
sys.path.insert(0, "/home/tham/Desktop/YOLOv8-multi-task-modified/ultralytics")
# 现在就可以导入Yolo类了
from ultralytics import YOLO

# Load a model
model = YOLO('/home/tham/Desktop/YOLOv8-multi-task-modified/ultralytics/models/v8/yolov8-bdd-v4-one-dropout-individual-n.yaml', task='multi').load('yolov8n.pt')  # build a new model from YAML
# model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
# model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

# Train the model
model.train(data='/home/tham/Desktop/YOLOv8-multi-task-modified/ultralytics/datasets/bdd-multi.yaml', batch=12, epochs=100, imgsz=(640,640), name='yolopm', val=True, task='multi',classes=[0,1,2,3,4,5],single_cls=False)

from ultralytics import YOLO


# Load a model
model = YOLO(r'D:\yolo\yolov10plus\ultralytics-main\ultralytics\cfg\models\v10\yolov10n.yaml')  # build a new model from YAML
model = YOLO(r'D:\yolo\yolov10plus\ultralytics-main\yolov10n.pt')  # load a pretrained model (recommended for training)
model = YOLO(r'D:\yolo\yolov10plus\ultralytics-main\ultralytics\cfg\models\v10\yolov10n.yaml').load(r'D:\yolo\yolov10plus\ultralytics-main\yolov10n.pt')  # build from YAML and transfer weights

# model_yaml_path = r'D:\yolo\yolov10plus\ultralytics-main\ultralytics\cfg\models\v10\yolov10n.yaml'
# 数据集配置文件
# data_yaml_path = r'D:\yolo\yolov10plus\ultralytics-main\data.yaml'
# 预训练模型
# pre_model_name = r'D:\yolo\yolov10plus\ultralytics-main\yolov10n.pt'

# Train the model
model.train(data='data.yaml',
            epochs=100,
            imgsz=640,
            batch=2,
            )
# workers = 10,
# patience = 20,
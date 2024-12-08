from ultralytics import YOLO

def main():
    # Load a model
    model = YOLO(r'D:\yolo\yolov10plus\ultralytics-main\ultralytics\cfg\models\v10\yolov10n.yaml')  # build a new model from YAML
    model = YOLO(r'D:\yolo\yolov10plus\ultralytics-main\yolov10n.pt')  # load a pretrained model (recommended for training)
    model = YOLO(r'D:\yolo\yolov10plus\ultralytics-main\ultralytics\cfg\models\v10\yolov10n.yaml').load(
        r'D:\yolo\yolov10plus\ultralytics-main\yolov10n.pt')  # build from YAML and transfer weights

    # Train the model
    model.train(data='data.yaml',
                epochs=100,

                batch=4)

if __name__ == '__main__':
    main()

# imgsz = 640,
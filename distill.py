import warnings
warnings.filterwarnings('ignore')
import argparse, yaml, copy
from ultralytics.models.yolo.detect.distill import DetectionDistiller
# from ultralytics.models.yolo.segment.distill import SegmentationDistiller
# from ultralytics.models.yolo.pose.distill import PoseDistiller
# from ultralytics.models.yolo.obb.distill import OBBDistiller

if __name__ == '__main__':
    param_dict = {
        # origin
        'model': 'ultralytics/cfg/models/11/yolo11-WTConv+SE.yaml',
        'data':'SSDD.yaml',
        'imgsz': 640,
        'epochs': 200,
        'batch': 32,
        'workers': 10,
        'cache': True,
        'optimizer': 'SGD',
        'device': '0',
        'close_mosaic': 20,
        # 'amp': False, # 如果蒸馏损失为nan，请把amp设置为False
        'project':'runs/distill',
        'name':'yolov11-chsim-exp1',
        
        # distill
        'prune_model': False,
        'teacher_weights': 'runs/detect/train11/weights/best.pt',
        'teacher_cfg': 'ultralytics/cfg/models/11/yolo11-WTConv+SE.yaml',
        'kd_loss_type': 'feature',
        'kd_loss_decay': 'constant',
        
        'logical_loss_type': 'BCKD',
        'logical_loss_ratio': 1.0,
        
        'teacher_kd_layers': '16, 19, 22',
        'student_kd_layers': '16, 19, 22',
        'feature_loss_type': 'cwd',
        'feature_loss_ratio': 1.0
    }
    
    model = DetectionDistiller(overrides=param_dict)
    # model = SegmentationDistiller(overrides=param_dict)
    # model = PoseDistiller(overrides=param_dict)
    # model = OBBDistiller(overrides=param_dict)
    model.distill()
import os
import glob
import json
import cv2
import pybboxes as pbx
import yaml
import argparse
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument("--config", help = "path of the training configuartion file", required = True)
args = parser.parse_args()

#Reading the configuration file
with open(args.config, 'r') as f:
    try:
        config = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(exc)

model = YOLO(config["model_path"])

if(config["gpu_avail"]):
    _ = model(source=config['images_path'],
            save=False,
            save_txt='True',
            conf=config['detection_conf_thresh'],
            device='cuda:0',
            name="yolo_images_pred")
else:
    _ = model(source=config['images_path'],
            save=False,
            save_txt='True',
            conf=config['detection_conf_thresh'],
            device='cpu',
            name="yolo_images_pred")


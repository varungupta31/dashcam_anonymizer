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
            save_txt=True,
            conf=config['detection_conf_thresh'],
            device='cuda:0',
            name="yolo_images_pred")
else:
    _ = model(source=config['images_path'],
            save=False,
            save_txt=True,
            conf=config['detection_conf_thresh'],
            device='cpu',
            name="yolo_images_pred")


#images = [int(item.split("/")[1].replace(config['img_format'], "")) for item in images]
images = sorted(glob.glob(config['images_path']+"/*"+config["img_format"]))

os.mkdir("annot_txt")

annot_dir = f'runs/detect/yolo_images_pred/labels/'

try:
    for file in os.listdir(annot_dir):
        if (file.endswith('.txt')):
            #frame_num = int(file.replace(".txt","").split("_")[1])
            with open(annot_dir+file, 'r') as fin:
                for line in fin.readlines():
                    line = [float(item) for item in line.split()[1:]]
                    line = pbx.convert_bbox(line, from_type="yolo", to_type="voc", image_size=(config["img_width"], config["img_height"]))
                    data_string = " ".join(str(num) for num in line)
                    with open(f"annot_txt/{os.path.basename(file)}", "a") as f:
                        f.write(data_string+"\n")
except:
    print(f'{os.path.basename(file)} has no detected objects.')

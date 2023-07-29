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

def sort_videos(item):
    return int(item.split("/")[1].replace(config['img_format'],""))

images = sorted(glob.glob("images/*"+config['img_format']), key=sort_videos)
images = [int(item.split("/")[1].replace(config['img_format'], "")) for item in images]


for image in images:
    cvimg = cv2.imread(image)
    W, H = cvimg.shape[1], cvimg.shape[0]
    data_dict = {}
    annot_dir = f'runs/detect/yolo_images_pred/labels/'
 
    try:
        for file in os.listdir(annot_dir):
            if (file.endswith('.txt')):
                frame_num = int(file.replace(".txt","").split("_")[1])
                with open(annot_dir+file, 'r') as fin:
                    for line in fin.readlines():
                        line = [float(item) for item in line.split()[1:]]
                        line = pbx.convert_bbox(line, from_type="yolo", to_type="voc", image_size=(W,H))
                        if(frame_num not in data_dict.keys()):
                            data_dict[frame_num] = [line]
                        data_dict[frame_num].append(line)
        if not os.path.exists("annot_jsons/"):
            os.makedirs("annot_jsons")                
        with open("annot_jsons/"+str(image)+".json", 'w') as f:
            json.dump(data_dict, f)
    except:
        print(f'{image} has no detected objects.')
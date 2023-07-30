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

with open(args.config, 'r') as f:
    try:
        config = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(exc)

model = YOLO(config["model_path"])

if(config["generate_detections"]):
    if(config["gpu_avail"]):
        _ = model(source=config['videos_path'],
                save=False,
                save_txt=True,
                conf=config['detection_conf_thresh'],
                device='cuda:0',
                name="yolo_vidoes_pred")
    else:
        _ = model(source=config['videos_path'],
                save=False,
                save_txt=True,
                conf=config['detection_conf_thresh'],
                device='cpu',
                name="yolo_videos_pred")
    

def sort_videos(item):
    return int(item.split("/")[1].replace(".mp4",""))

videos = sorted(glob.glob(f"{config['videos_path']}/*.mp4"), key=sort_videos)
videos = [int(item.split("/")[1].replace(".mp4", "")) for item in videos]

for video in videos:
    data_dict = {}
    annot_dir = glob.glob(f'runs/detect/yolo_videos_pred/labels/{video}_*.txt')
    try:
        for file in annot_dir:
            #file = os.path.basename(file)
            if (os.path.basename(file).endswith('.txt')):
                frame_num = int(os.path.basename(file).replace(".txt","").split("_")[1])
                with open(file, 'r') as fin:
                    for line in fin.readlines():
                        line = [float(item) for item in line.split()[1:]]
                        line = pbx.convert_bbox(line, from_type="yolo", to_type="voc", image_size=(config["vid_width"],config["vid_height"]))
                        if(frame_num not in data_dict.keys()):
                            data_dict[frame_num] = [line]
                        data_dict[frame_num].append(line)
        if(not os.path.exists("annot_jsons/")):
            os.mkdir("annot_jsons")
        with open("annot_jsons/"+str(video)+".json", 'w') as f:
            json.dump(data_dict, f)
    except:
        print(f'{video} has no Annotation!')
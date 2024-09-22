import os
import glob
import json
import cv2
import pybboxes as pbx
import yaml
import argparse
from ultralytics import YOLO
import shutil
from rich.console import Console
from rich.progress import track
from natsort import natsorted
from os.path import join as osj

parser = argparse.ArgumentParser()
parser.add_argument("--config", help = "path of the training configuartion file", required = True)
args = parser.parse_args()
console = Console()


console.print(f"Reading the Configuration file from {args.config}", style="bold green")
with open(args.config, 'r') as f:
    try:
        config = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(exc)


console.print("Loading YOLO Model...", style="bold green")
model = YOLO(config["model_path"])

if(config["generate_detections"]):
    console.print("Generating YOLO Detections for the Videos", style="bold green")
    if(config["gpu_avail"]):
        console.print("GPU Available, Running on GPU", style="bold green")
        _ = model(source=config['videos_path'],
                save=False,
                save_txt=True,
                conf=config['detection_conf_thresh'],
                device='cuda:0',
                project='runs/detect/',
                name="yolo_videos_pred")
    else:
        console.print("GPU Not Available, Running on CPU", style="bold orange")
        _ = model(source=config['videos_path'],
                save=False,
                save_txt=True,
                conf=config['detection_conf_thresh'],
                device='cpu',
                project='runs/detect/',
                name="yolo_videos_pred")
    

videos = natsorted(glob.glob(f"{config['videos_path']}/*.mp4"))

if(config["generate_jsons"]):
    print(f"Generating JSONs for {len(videos)} videos")
    for video in track(videos):
        #finding video dimensions
        vid_name = os.path.basename(video).replace(".mp4","")
        vid = cv2.VideoCapture(video)
        height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        
        data_dict = {}
        annot_dir = natsorted(glob.glob(f'runs/detect/yolo_videos_pred/labels/{vid_name}_*.txt'))
        
        try:
            for file in annot_dir:
                #file = os.path.basename(file)
                if (os.path.basename(file).endswith('.txt')):
                    frame_num = int(os.path.basename(file).replace(".txt","").split("_")[1])
                    with open(file, 'r') as fin:
                        for line in fin.readlines():
                            line = [float(item) for item in line.split()[1:]]
                            line = pbx.convert_bbox(line, from_type="yolo", to_type="voc", image_size=(width,height))
                            if(frame_num not in data_dict.keys()):
                                data_dict[frame_num] = [line]
                            data_dict[frame_num].append(line)
            if(not os.path.exists("annot_jsons/")):
                os.mkdir("annot_jsons")
            with open("annot_jsons/"+str(vid_name)+".json", 'w') as f:
                json.dump(data_dict, f)
        except:
            print(f'{video} has no Annotation!')

#Annotation JSONS stored in annot_jsons folder.

def blur_regions(image, regions):
    """
    Blurs the image, given the x1,y1,x2,y2 cordinates using Gaussian Blur.
    15,15 gaussian radius enough to anonymyze (may increase if needed)
    """
    for region in regions:
        x1,y1,x2,y2 = region
        x1, y1, x2, y2 = round(x1), round(y1), round(x2), round(y2)
        roi = image[y1:y2, x1:x2]
        blurred_roi = cv2.GaussianBlur(roi, (config["blur_radius"], config["blur_radius"]), 0)
        image[y1:y2, x1:x2] = blurred_roi
    return image

#videos = glob.glob(config["videos_path"]+"*.mp4")


if not(os.path.exists(config["output_folder"])):
    console.print(f"Creating Directory {config['output_folder']} to store the anonymized videos", style="bold green")
    os.mkdir(config["output_folder"])

anonymized_videos_path = config["output_folder"]

for video in track(videos):
    vid_name = os.path.basename(video).replace(".mp4","")
    json_path = f'annot_jsons/{vid_name}.json'
    if(os.path.exists(json_path)):
        with open(json_path) as F:
            #Data is the json dictionary in which key is the frame, and value is a list of lists.
            data = json.load(F)

            #video writer setup
            video_capture = cv2.VideoCapture(video)
            out_vid_path = osj(anonymized_videos_path, vid_name+'.mp4')
            
            frame_width = int(video_capture.get(3))
            frame_height = int(video_capture.get(4))
            frame_size = (frame_width,frame_height)
            
            fps = round(video_capture.get(cv2.CAP_PROP_FPS))
            output_video = cv2.VideoWriter(out_vid_path, cv2.VideoWriter_fourcc(*'avc1'), fps, frame_size)
            count = 1
            while True:
                ret, frame = video_capture.read()
                
                if not ret:
                    break
                try:
                    #the frame has some detection
                    frame = blur_regions(frame, data[str(count)])
                except:
                    #otherwise write the frame as is
                    frame = frame
                output_video.write(frame)
                count+=1
            video_capture.release()
            output_video.release()
        print(f"Processed Video {vid_name}")
    else:
        console.print(f"No objects in file {video}, copying file as is. PLEASE CHECK DETECTOR AGAIN.", style="bold red")
        shutil.copy(video, anonymized_videos_path)
        console.print(f"Processed Video {vid_name}", style="bold green")
        
#remove runs folder
console.print(f"Removing Temporary Files..")
shutil.rmtree("runs/")
shutil.rmtree("annot_jsons/")

console.print(f"Blurred Videos are stored in {out_vid_path}", style="bold yellow")
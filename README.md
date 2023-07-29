<h1> Dashcam Anonymizer </h1>

This repository blurs human faces and license plates in images and videos, using a state-of-the-art object detection model, [YOLOv8 by Ultralytics](https://github.com/ultralytics/ultralytics) and is fine-tuned using images from the [OpenImagesDatasetV7](https://storage.googleapis.com/openimages/web/index.html).

git clone this repo by
```
https://github.com/varungupta31/dashcam_anonymizer.git
```

Install the conda environment

```
conda env create -f environment.yaml
```

<h3> Blurring Images in a Directory </h3>

To blur all images in a directory,

Update the `configs/img_blur.yaml` as required, and run the following command

```
python blur_images.py --config configs/img_blur.yaml
```
The resulting blur images will be stored in the directory specified in the YAML.
Note: `annot_txt` folder will contain the YOLO detections in `.txt` format, converted to the `VOC` bounding-box format.

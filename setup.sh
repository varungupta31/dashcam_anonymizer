pip install ultralytics==8.0.144
pip install pybboxes
pip install opencv-python
conda install opencv=4.6.0
pip uninstall numpy
pip install numpy==1.25.1
pip install gdown
mkdir model
echo "Downloading the YOLO model..."
gdown 1uV8IMuGDbmDabdjyeSy4SUKV9OS-ULbe
mv best.pt model/
echo "Setup complete!"
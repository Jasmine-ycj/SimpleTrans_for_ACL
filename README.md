# SimpleTrans_for_ACL
## Images & Labels
- Images for classification (alreagy cropped according to the extracted ROI):

`./data_label/images/crops/ACL`

- Bounding box annotation on 246 cases:

`./data_label/labels_for_YOLO/all_YOLO_labels.zip`

- Case number & ACL tear severity labels of 663 cases (0- intact, 1- partial tear, 2- full tear):

`./data_label/labels_for_classification/clc663_3.csv` 

## The trained network
- Our trained YOLOv5m:

`./trained_nets/YOLO/best.pt`

- Our trained classification network:

`./trained_nets/classification/`
## Results
- ACL Localization results on 663 cases:

`./results/YOLO/labels/`

- ACL tear classification results in 5-fold cross validating:

`./results/classification/`

## Running for ACL Tear Classification
- Train.py: 5-fold cross training & validating

- Val.py: 5-fold cross validating
## Running for ACL Localization
In the [YOLOv5 project](https://github.com/ultralytics/yolov5), use `./trained_nets/YOLO/best.pt` as weights and replace the original `detect.py` with `detect_ACL.py` to run.

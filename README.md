# SimpleTrans_for_ACL
## Paper
YU C, WANG M, CHEN S, et al. Improving anterior cruciate ligament tear detection and grading through efficient use of inter-slice information and simplified transformer module[J]. Biomedical Signal Processing and Control, 2023, 86: 105356.

https://doi.org/10.1016/j.bspc.2023.105356.

## Abstract
Deep learning has shown considerable promise in magnetic resonance imaging (MRI)-based anterior cruciate ligament (ACL) tear detection. However, tear severity grading still presents challenges, including the low sensitivity for partial tears and the absence of convenient methods for region of interest (ROI) extraction. To address these issues, our study proposes a solution by effectively utilizing inter-slice information. For the classification task, we employ a two-dimensional (2D) backbone network for independent feature extraction, along with a simplified transformer module for integrating inter-layer information. Additionally, a post-processing method is developed for You Only Look Once (YOLO)-based localization, leveraging the distribution characteristics of the ACL in MR scans. We conduct experiments on the Osteoarthritis Initiative (OAI) dataset, and the results demonstrate a significantly improved sensitivity of 0.9222 for partial tears. Additionally, our classification network outperforms both MRNet and a 3D network in terms of overall performance while maintaining a compact size of 701 kB. Moreover, the advantages of our network and the necessity of transformer simplification are validated through class activation mapping (CAM) and ablation experiments, respectively. With appropriate inter-slice integration methods, our model is efficient and lightweight. Our approach successfully transformed and applied the transformer in ACL tear grading and significantly enhanced the accuracy. The streamlined and lightweight model is also prepared for future deployment in clinical settings or migration to mobile devices.
## Images & Labels
- JPG images after preprocessing:
  
`./data_label/images/OAI-JPG.zip`

- Images for classification (alreagy cropped according to the extracted ROI):

`./data_label/images/crops/ACL`

- Bounding box annotations on 246 cases:

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

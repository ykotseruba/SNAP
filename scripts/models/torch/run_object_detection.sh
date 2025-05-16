#!/bin/bash
set -x
DATA_PATH=data
RESULTS_PATH=../../../raw_results/object_detection/

for model_name in SSD300_VGG16 SSDLite320_MobileNet_V3_Large FasterRCNN_MobileNet_V3_Large_320_FPN FasterRCNN_MobileNet_V3_Large_FPN FasterRCNN_ResNet50_FPN RetinaNet_ResNet50_FPN
do
python3 detect.py --model_name $model_name --results_path  $RESULTS_PATH/${model_name}.json
done

# for model_name in deta-resnet-50 deta-swin-large detr-resnet-50 detr-resnet-101
# do
# 	python3 detect.py --model_name $model_name --results_path  $RESULTS_PATH/${model_name}.json
# done

# for model_name in owlvit-large-patch14 grounding-dino-base 
# do
# 	python3 detect.py --model_name $model_name --results_path ${RESULTS_PATH}/${model_name}.json
# done

# for model_name in dino-r50 dino-swin-l
# do
#  	python3 detect.py --model_name $model_name --results_path ${RESULTS_PATH}/${model_name}.json
# done

# for model_name in cascade_mask_rcnn_vitdet_b cascade_mask_rcnn_vitdet_l cascade_mask_rcnn_vitdet_h cascade_mask_rcnn_swin_l
# do
#  	python3 detect.py --model_name $model_name --results_path  $RESULTS_PATH/${model_name}.json
# done

# for model_name in mask_rcnn_vitdet_b  mask_rcnn_vitdet_h mask_rcnn_vitdet_l cascade_mask_rcnn_swin_b
# do
#   	python3 detect.py --model_name $model_name --results_path  $RESULTS_PATH/${model_name}.json
# done

# # to run these first
# # pip3 install torch==2.2.0 transformers==4.51.3
# for model_name in rtdetr_v2_r101vd rtdetr_v2_r50vd rtdetr_r101vd rtdetr_r50vd rtdetr_r50vd_coco_o365 rtdetr_r101vd_coco_o365 
# do
# 	python3 -m pdb detect.py --model_name $model_name --results_path  $RESULTS_PATH/${model_name}.json
# done
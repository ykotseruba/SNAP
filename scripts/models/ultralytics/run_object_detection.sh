#!/bin/bash
set -x

for model_name in yolo12x rtdetr-l yolov3 yolov5x yolov8x yolov10x rtdetr-x;
do
	python3 run_model.py --model_name $model_name --task detect --input_path ../../../SNAP/ --results_path ../../../raw_results/object_detection/${model_name}.json
done 

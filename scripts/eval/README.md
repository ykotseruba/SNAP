# Evaluation

There are scripts for evaluating each task:

- `eval_image_classification.py`
- `eval_object_detection.py`
- `eval_vqa.py`

Each script should be run from the root directory (`SNAP`), which will produce excel file with model results for each image and grouped by EV_offset and other attributes. 

```
python3 scripts/eval/eval_image_classification.py
```

Individual model results will be saved in `eval_results/<task_dir>`, where task_dir is the vision task, e.g. image_classification.

Another excel file `all_models.xlsx` will be created in the `eval_results/<task_dir>`, which aggregates results over all tested models. 

`generate_paper_figures.py` generates all figures in the paper and appendices.

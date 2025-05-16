## Run image classification and object detection models

### Build docker image

```
bash docker/build_docker.sh
```

### Run docker image
Before running docker modify paths to DATA and CODE_FOLDER in the run_docker.sh file.
Then run the following command:

```
bash docker/run_docker.sh
```

Inside docker run image classification and object detection models on SNAP:

```
./run_image_classification.sh
./run_object_detection.sh
```

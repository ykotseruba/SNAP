DATA=$SNAP_DATA_PATH
WORKING_DIR=$SNAP_PATH

xhost +local:docker
docker run --ipc=host \
--gpus "device=0" \
--mount type=bind,source=$WORKING_DIR,target=/ultralytics/SNAP \
-e SENSOR_BIAS_PATH=$SENSOR_BIAS_PATH \
-v /tmp/.X11-unix:/tmp/.X11-unix \
-v ~/.Xauthority:/root/.Xauthority \
-it ultralytics/ultralytics:latest
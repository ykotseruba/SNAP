#!/bin/bash
set -e
# setup x auth environment for visual support
XAUTH=$(mktemp /tmp/.docker.xauth.XXXXXXXXX)
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -

###################################################################
########### UPDATE PATHS BELOW BEFORE RUNNING #####################
###################################################################

# Provide full path sensor bias code directory
CODE_FOLDER=$SNAP_PATH

# Provide full path to image data
SENSOR_BIAS_DATA=$SNAP_DATA_PATH


###################################################################
########### DO NOT MODIFY SETTINGS BELOW ##########################
##### CHANGE DEFAULT DOCKER IMAGE NAME, TAG, GPU DEVICE, ##########
########## MEMORY LIMIT VIA COMMAND LINE PARAMETERS ###############
###################################################################


IMAGE_NAME=base_images/pytorch
TAG=paligemma
CONTAINER_NAME=paligemma

# DOCKER TEMP
KERAS_TEMP=/tmp/.keras
DOCKER_TEMP=$HOME/dockers/docker_temp

WORKING_DIR=$(dirname "$(readlink -f "${BASH_SOURCE}")")/..

# gpu and memory limit
GPU_DEVICE=1
MEMORY_LIMIT=32g

# options
INTERACTIVE=1
LOG_OUTPUT=1

while [[ $# -gt 0 ]]
do key="$1"

case $key in
	-im|--image_name)
	IMAGE_NAME="$2"
	shift # past argument
	shift # past value
	;;
	-t|--tag)
	TAG="$2"
	shift # past argument
	shift # past value
	;;
	-i|--interactive)
	INTERACTIVE="$2"
	shift # past argument
	shift # past value
	;;
	-gd|--gpu_device)
	GPU_DEVICE="$2"
	shift # past argument
	shift # past value
	;;
	-m|--memory_limit)
	MEMORY_LIMIT="$2"
	shift # past argument
	shift # past value
	;;
	-cn|--container_name)
	CONTAINER_NAME="$2"
	shift # past argument
	shift # past value
	;;
	-h|--help)
	shift # past argument
	echo "Options:"
	echo "	-im, --image_name 	name of the docker image (default \"base_images/tensorflow\")"
	echo "	-t, --tag 		image tag name (default \"tf2-gpu\")"
	echo "	-gd, --gpu_device 	gpu to be used inside docker (default 1)"
	echo "	-cn, --container_name	name of container (default \"tf2_run\" )"
	echo "	-m, --memory_limit 	RAM limit (default 32g)"
	exit
	;;
	*)
	echo " Wrong option(s) is selected. Use -h, --help for more information "
	exit
	;;
esac
done

echo "GPU_DEVICE 	= ${GPU_DEVICE}"
echo "CONTAINER_NAME 	= ${CONTAINER_NAME}"


echo "Running docker in interactive mode"

# docker flags
# -rm  		- option erases container after completion 
# -it  		- interactive, allows to run commands inside container
# --gpus 	- which device to use
# --mount 	- type=bind bind mount allows to mount file/folder from host to container
#			source - absolute path on the host machine
#			target - relative path inside docker, if folder does not exist, it will be created
# -m 		- max RAM available to docker
# -w 		working directory inside the container
# -e 		environment variables
# -v 		volume, similar to mount, but is managed by the docker
# -p 		port
# -name 	name of the container
# Xauthority, DISPLAY, etc are needed for X11 forwarding, e.g. to use GUI or show plots
# UID, GID variables set up user and group id inside docker

	# --mount type=bind,source=${HOME}/.cache/,target=/.cache \
	# --mount type=bind,source=${HOME}/.keras/,target=/home/docker_user/.keras \ 
	#--mount type=bind,source=${SENSOR_BIAS_DATA},target=${CODE_FOLDER}/data_v3 \

docker run --rm -it --gpus \"device=${GPU_DEVICE}\"  \
	--mount type=bind,source=${HOME}/.cache/,target=/home/docker_user/.cache \
	--mount type=bind,source=${CODE_FOLDER},target=${CODE_FOLDER} \
	-e SENSOR_BIAS_DATA=${CODE_FOLDER}/data/ \
	-m ${MEMORY_LIMIT} \
	-w ${WORKING_DIR} \
	-e log=/home/log.txt \
	-e HOST_UID=$(id -u) \
	-e HOST_GID=$(id -g) \
	-u $(id -u):$(id -g) \
	-e DISPLAY=$DISPLAY \
	-e XAUTHORITY=$XAUTH \
	-v $XAUTH:$XAUTH \
	-p 8008:6006 \
   	--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
   	--ipc=host \
	--name ${CONTAINER_NAME} \
	--net=host \
	-env="DISPLAY" \
	--volume="$HOME/.Xauthority:/root/.Xauthority:rw" \
	${IMAGE_NAME}:${TAG}

FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

ENV NVIDIA VISIBLE_DEVICES\
	${NVIDIA_VISIBLE_DEVICES:-all}

ENV NVIDIA_DRIVER_CAPABILITIES\
	${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64/

ENV PYTHONPATH=/home/iuliia/Documents/sensor_bias/scripts/models/vqa_models/llava/LLaVA:$PYTHONPATH

SHELL ["/bin/bash", "-c"]

EXPOSE 6006 6007 6008

ARG USER_ID
ARG GROUP_ID

RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID docker_user


ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update&& apt-get install -y -q \
	vim \
	gosu \
	g++ \
	gedit \
	curl \
	unzip \
	git \
	wget \
	python3-pip \
	python3-testresources \
	libcanberra-gtk-module \	
 	libcanberra-gtk3-module

COPY requirements.txt /
RUN pip3 install -r /requirements.txt

RUN ln -s /usr/bin/python3 /usr/bin/python

USER docker_user
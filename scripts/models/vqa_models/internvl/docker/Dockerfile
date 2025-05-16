FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel
#FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04


ENV NVIDIA VISIBLE_DEVICES\
	${NVIDIA_VISIBLE_DEVICES:-all}

ENV NVIDIA_DRIVER_CAPABILITIES\
	${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64/

SHELL ["/bin/bash", "-c"]

EXPOSE 6006 6007 6008

ARG USER_ID
ARG GROUP_ID

RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID docker_user


ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update&& apt-get install -y -q graphviz libgraphviz-dev \
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

# not supported on Titan GPU
#RUN pip3 install ninja
#RUN MAX_JOBS=4 pip3 install flash-attn==2.3.6 --no-build-isolation

RUN pip3 install -r /requirements.txt

RUN ln -s /usr/bin/python3 /usr/bin/python

USER docker_user
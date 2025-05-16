FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

ARG HOME
ARG USER_ID
ARG GROUP_ID
ARG PYTHONPATH

ENV NVIDIA VISIBLE_DEVICES\
	${NVIDIA_VISIBLE_DEVICES:-all}

ENV NVIDIA_DRIVER_CAPABILITIES\
	${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64/

SHELL ["/bin/bash", "-c"]

EXPOSE 6006 6007 6008

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
RUN pip3 install -r /requirements.txt

# without ninja flash attention builds forever
RUN pip3 install ninja
RUN MAX_JOBS=4 pip3 install flash-attn==1.0.3.post0 --no-build-isolation

#RUN pip3 install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

#RUN site_pkg_path=$(python3 -c 'import site; print(site.getsitepackages()[0])')
#RUN cp -rv ../VILA/llava/train/transformers_replace/* $site_pkg_path/transformers/
#RUN cp -rv ../llava/train/deepspeed_replace/* $site_pkg_path/deepspeed/

RUN ln -s /usr/bin/python3 /usr/bin/python

ENV PYTHONPATH=$HOME/Documents/sensor_bias/scripts/models/vqa_models/vila/VILA/llava:$PYTHONPATH

USER docker_user
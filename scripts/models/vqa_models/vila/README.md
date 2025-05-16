# VILA

1. clone the repo
git clone https://github.com/NVlabs/VILA

2. build the docker
docker/build_docker.sh 

3. run the docker on two GPUs (one is not enough)
docker/run_docker.sh -gd 2,3

4. move run_vila.py inside VILA directory

5. inside the docker run
python3 run_llava.py <model_name>

Available model names
VILA-2.7b
VILA-7b
VILA1.5-3b
VILA1.5-7b

Note: the original code for VILA1.5-7b does not run on multiple GPUs 

line 151 in llava/model/builder.py -- change kwargs to device_map='auto'
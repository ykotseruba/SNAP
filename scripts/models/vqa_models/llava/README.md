# VILA

1. clone the repo
git clone https://github.com/haotian-liu/LLaVA.git

2. build the docker
docker/build_docker.sh 

3. run the docker on two GPUs (one is not enough)
docker/run_docker.sh -gd 2,3

4. inside the docker run
python3 run_llava.py liuhaotian/llava-v1.6-vicuna-7b

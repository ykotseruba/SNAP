# Running VLMs on SNAP

The procedure is similar for all vlms.

1. Build the docker

`docker/build_docker.sh`

2. Run the docker

`docker/run_docker.sh`

If more than one GPU is needed:

`docker/run_docker.sh -gd 2,3`

3. Inside the docker run the script for the corresponding vlm

`python3 run_<vlm>.py <model_name>`

See READMEs for addional running instructions for individual VLMs
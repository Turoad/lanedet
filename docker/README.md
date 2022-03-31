# Building a LaneDet container

### Install Docker and NVIDIA Container Toolkit

https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

### Build Container

> 

```Shell
# From root of LaneDet repo
cd $LANEDET_ROOT

# Build:
docker build -f docker/Dockerfile -t lanedet:latest .

# Run:
docker run --gpus all -it \
	--shm-size=8gb  \
	--name=lanedet --ipc=host --net=host lanedet:latest
```
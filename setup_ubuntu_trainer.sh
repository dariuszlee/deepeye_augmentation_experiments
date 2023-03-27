#!/bin/bash
## This gist contains instructions about cuda v11.2 and cudnn8.1 installation in Ubuntu 20.04 for Pytorch 1.8 & Tensorflow 2.7.0


### If you have previous installation remove it first. 
sudo apt-get purge -y nvidia*
sudo apt remove -y nvidia-*
sudo rm /etc/apt/sources.list.d/cuda*
sudo apt-get -y autoremove && sudo apt-get -y autoclean
sudo rm -rf /usr/local/cuda*


### to verify your gpu is cuda enable check
lspci | grep -i nvidia

### gcc compiler is required for development using the cuda toolkit. to verify the version of gcc install enter
gcc --version

# system update
sudo apt-get update -y
sudo apt-get upgrade -y
sudo apt-get install -y g++ freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev

# first get the PPA repository driver
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update -y

# install nvidia driver with dependencies
sudo apt install -y libnvidia-common-470 libnvidia-gl-470 nvidia-driver-470 --allow-unauthenticated


sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64 /" | sudo tee /etc/apt/sources.list.d/cuda.list
sudo apt-get update -y --allow-unauthenticated --allow-insecure-repositories

 # installing CUDA-11.2
sudo apt install -y cuda-11-2 --allow-unauthenticated


# setup your paths
echo 'export PATH=/usr/local/cuda-11.2/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.2/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
sudo ldconfig

# install cuDNN v8.1
# in order to download cuDNN you have to be regeistered here https://developer.nvidia.com/developer-program/signup
# then download cuDNN v8.1 form https://developer.nvidia.com/cudnn

CUDNN_TAR_FILE="cudnn-11.2-linux-x64-v8.1.1.33.tgz"
tar -xzvf ${CUDNN_TAR_FILE}

# copy the following files into the cuda toolkit directory.
sudo cp -P cuda/include/cudnn.h /usr/local/cuda-11.2/include
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda-11.2/lib64/
sudo chmod a+r /usr/local/cuda-11.2/lib64/libcudnn*

rm cuda -rf
rm ${CUDNN_TAR_FILE}

# Finally, to verify the installation, check
nvidia-smi
nvcc -V

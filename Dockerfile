FROM python:3.8-buster

COPY ./requirements.txt /
RUN pip install -r requirements.txt

COPY ./Data/ /Data
COPY ./Evaluation/ /Evaluation
COPY ./configs/ /configs
COPY ./Model/ /Model
COPY ./DataAugmentation /DataAugmentation
COPY ./evaluate_multiple.py /

RUN apt update -y
RUN apt-get install -y software-properties-common
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/debian10/x86_64/7fa2af80.pub
RUN add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/debian10/x86_64/ /"
RUN add-apt-repository contrib
RUN apt-get --allow-unauthenticated --allow-insecure-repositories update

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get --allow-unauthenticated -y install cuda

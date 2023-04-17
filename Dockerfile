# FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime 
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

ENV  PYTHONFAULTHANDLER=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONHASHSEED=random \
  PIP_NO_CACHE_DIR=off \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  PIP_DEFAULT_TIMEOUT=100

# Prevent stop building ubuntu at time zone selection.  
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update --fix-missing
RUN apt-get -y install --no-install-recommends apt-utils dialog 2>&1
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install -y --fix-missing protobuf-compiler \
                       libsm6 \
                       libxext6 \
                       libxrender-dev \
                       wget make curl unzip git vim bash-completion locales build-essential

WORKDIR /work

COPY ./requirements.txt /work/requirements.txt
COPY ./Makefile /work/Makefile

RUN make install